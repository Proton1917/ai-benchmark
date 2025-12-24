"""
AI 数学 Benchmark 系统 - 后端服务
"""

import json
import os
import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import re

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
QUESTIONS_FILE = DATA_DIR / "questions.json"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# SQLite 数据库
DB_PATH = DATA_DIR / "stats.db"

# FastAPI 应用
app = FastAPI(title="AI Math Benchmark")

# 评测进度（仅内存）
EVAL_PROGRESS = {}


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS question_stats (
            question_id TEXT PRIMARY KEY,
            correct_count INTEGER NOT NULL DEFAULT 0,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_eval (
            eval_id TEXT NOT NULL,
            model TEXT NOT NULL,
            total_score REAL NOT NULL,
            max_score REAL NOT NULL,
            score_rate REAL NOT NULL,
            normalized_total_score REAL NOT NULL,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (eval_id, model)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ui_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def update_question_stats(question_id: str, is_correct: bool):
    now = datetime.now().isoformat()
    correct_inc = 1 if is_correct else 0
    attempt_inc = 1
    conn = db_connect()
    conn.execute("""
        INSERT INTO question_stats (question_id, correct_count, attempt_count, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(question_id) DO UPDATE SET
            correct_count = correct_count + ?,
            attempt_count = attempt_count + ?,
            last_updated = ?
    """, (question_id, correct_inc, attempt_inc, now, correct_inc, attempt_inc, now))
    conn.commit()
    conn.close()


def get_question_stats_map() -> dict:
    conn = db_connect()
    rows = conn.execute("SELECT question_id, correct_count, attempt_count FROM question_stats").fetchall()
    conn.close()
    stats = {}
    for qid, correct, attempt in rows:
        stats[qid] = {
            "correct_count": correct,
            "attempt_count": attempt,
        }
    return stats


def record_model_eval(eval_id: str, model: str, total_score: float, max_score: float, score_rate: float, normalized_total_score: float):
    conn = db_connect()
    conn.execute("""
        INSERT OR REPLACE INTO model_eval (eval_id, model, total_score, max_score, score_rate, normalized_total_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (eval_id, model, total_score, max_score, score_rate, normalized_total_score, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def record_model_eval_with_ts(eval_id: str, model: str, total_score: float, max_score: float, score_rate: float, normalized_total_score: float, timestamp: str):
    conn = db_connect()
    conn.execute("""
        INSERT OR REPLACE INTO model_eval (eval_id, model, total_score, max_score, score_rate, normalized_total_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (eval_id, model, total_score, max_score, score_rate, normalized_total_score, timestamp))
    conn.commit()
    conn.close()


def get_ui_state_db() -> dict:
    conn = db_connect()
    rows = conn.execute("SELECT key, value FROM ui_state").fetchall()
    conn.close()
    data = {"tested_models": [], "judge_model": "openai/gpt-5.2", "question_ids": []}
    for key, value in rows:
        try:
            if key in ("tested_models", "question_ids"):
                data[key] = json.loads(value)
            else:
                data[key] = value
        except Exception:
            data[key] = value
    return data


def set_ui_state_db(state) -> dict:
    data = state.model_dump()
    conn = db_connect()
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        conn.execute("INSERT OR REPLACE INTO ui_state (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()
    return data


init_db()


def migrate_results_to_current_scores():
    questions = load_questions()
    score_map = {q["id"]: q.get("score", 0) for q in questions.get("questions", [])}
    # 先清空历史统计，重建
    conn = db_connect()
    conn.execute("DELETE FROM model_eval")
    conn.execute("DELETE FROM question_stats")
    conn.commit()
    conn.close()

    stats = {}

    for file in sorted(RESULTS_DIR.glob("eval_*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        changed = False
        for model, info in data.get("results", {}).items():
            total_score = 0.0
            max_score = 0.0
            for d in info.get("details", []):
                qid = d.get("question_id")
                new_qscore = score_map.get(qid, d.get("question_score", 0))
                if d.get("question_score") != new_qscore:
                    d["question_score"] = new_qscore
                    changed = True
                weight = d.get("score_ratio")
                if weight is None:
                    weight = 1.0 if d.get("is_correct") else 0.0
                new_earned = float(new_qscore) * float(weight)
                if d.get("earned_score") != new_earned:
                    d["earned_score"] = new_earned
                    changed = True
                total_score += new_earned
                max_score += float(new_qscore)

                # 统计正确率（满分为正确）
                if qid:
                    st = stats.setdefault(qid, {"correct": 0, "attempt": 0})
                    st["attempt"] += 1
                    if weight >= 1.0:
                        st["correct"] += 1

            info["total_score"] = total_score
            info["max_score"] = max_score
            info["score_rate"] = (total_score / max_score) if max_score > 0 else 0
            info["normalized_total_score"] = info["score_rate"] * 100

            record_model_eval_with_ts(
                data.get("id"),
                model,
                info["total_score"],
                info["max_score"],
                info["score_rate"],
                info["normalized_total_score"],
                data.get("timestamp") or datetime.now().isoformat(),
            )

        if changed:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # 写回题目统计
    conn = db_connect()
    for qid, st in stats.items():
        conn.execute("""
            INSERT INTO question_stats (question_id, correct_count, attempt_count, last_updated)
            VALUES (?, ?, ?, ?)
        """, (qid, st["correct"], st["attempt"], datetime.now().isoformat()))
    conn.commit()
    conn.close()


# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter API 配置
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("缺少环境变量 OPENROUTER_API_KEY，请先在本地设置后再启动服务。")
MAX_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "20"))
OPENROUTER_HEALTH = {"ok": None, "checked_at": 0, "latency_ms": None}
OPENROUTER_HEALTH_TTL = 10


# ==================== 数据模型 ====================

class Question(BaseModel):
    id: str
    question: str
    answer: Union[float, str]
    score: float
    tolerance: Optional[float] = 0
    answer_text: Optional[str] = None
    exact_required: Optional[bool] = True
    judge_hint: Optional[str] = None


class QuestionCreate(BaseModel):
    question: str
    answer: Union[float, str]
    score: float
    tolerance: Optional[float] = 0
    answer_text: Optional[str] = None
    exact_required: Optional[bool] = True
    judge_hint: Optional[str] = None


class EvaluateRequest(BaseModel):
    tested_models: list[str]
    judge_model: str = "openai/gpt-5.2"
    question_ids: list[str]


class UiState(BaseModel):
    tested_models: list[str] = []
    judge_model: str = "openai/gpt-5.2"
    question_ids: list[str] = []



# ==================== 辅助函数 ====================

def load_questions() -> dict:
    """加载题库"""
    if QUESTIONS_FILE.exists():
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"questions": []}


def save_questions(data: dict):
    """保存题库"""
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_question_id() -> str:
    """生成题目ID"""
    data = load_questions()
    existing_ids = {q["id"] for q in data["questions"]}
    i = 1
    while f"q{i:03d}" in existing_ids:
        i += 1
    return f"q{i:03d}"

migrate_results_to_current_scores()


async def call_openrouter(
    api_key: str,
    model: str,
    messages: list,
    timeout: float = 120,
    extra_params: Optional[dict] = None,
    drop_params: Optional[list[str]] = None,
):
    """调用 OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
    }
    if extra_params:
        payload.update(extra_params)
    if drop_params:
        for key in drop_params:
            payload.pop(key, None)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return content, usage


async def call_openrouter_stream(
    api_key: str,
    model: str,
    messages: list,
    timeout: float = 120,
    extra_params: Optional[dict] = None,
    drop_params: Optional[list[str]] = None,
    on_chunk=None,
):
    """流式调用 OpenRouter，返回完整内容与最终 usage"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_params:
        payload.update(extra_params)
    if drop_params:
        for key in drop_params:
            payload.pop(key, None)

    content = ""
    usage = {}
    saw_data = False
    fallback_lines = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=await response.aread())

            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    fallback_lines.append(line)
                    continue
                saw_data = True
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # delta 内容
                delta = ""
                if "choices" in obj and obj["choices"]:
                    choice = obj["choices"][0]
                    if "delta" in choice and isinstance(choice["delta"], dict):
                        delta = choice["delta"].get("content") or ""
                    elif "message" in choice and isinstance(choice["message"], dict):
                        delta = choice["message"].get("content") or ""
                if delta:
                    content += delta
                    if on_chunk:
                        await on_chunk(delta)
                if "usage" in obj:
                    usage = obj.get("usage") or usage

            if not saw_data and fallback_lines:
                raw = "\n".join(fallback_lines).strip()
                try:
                    obj = json.loads(raw)
                    if "choices" in obj and obj["choices"]:
                        choice = obj["choices"][0]
                        if "message" in choice and isinstance(choice["message"], dict):
                            content = choice["message"].get("content") or ""
                        elif "delta" in choice and isinstance(choice["delta"], dict):
                            content = choice["delta"].get("content") or ""
                    if "usage" in obj:
                        usage = obj.get("usage") or usage
                except json.JSONDecodeError:
                    pass
                if content and on_chunk:
                    await on_chunk(content)

    return content, usage


def parse_extracted_answer(judge_response: str):
    """解析裁判返回的 extracted_answer，容错 JSON 失效的情况"""
    try:
        result = json.loads(judge_response)
        return result.get("extracted_answer")
    except json.JSONDecodeError:
        # 容错提取：尽量抓取 extracted_answer 的值
        match = re.search(r'"extracted_answer"\\s*:\\s*(.+)', judge_response)
        if not match:
            return None
        value = match.group(1).strip()
        # 去掉结尾多余内容
        value = re.sub(r'\\s*}\\s*$', '', value)
        value = re.sub(r',\\s*$', '', value)
        if value.lower().startswith("null"):
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value


def build_judge_prompt(question: str, model_response: str) -> str:
    """构建裁判 AI 的 Prompt - 提取最终答案（保留精确形式）"""
    return f"""你是一个数学题答案提取专家。请从被测AI的回答中提取最终答案，尽量保持精确形式（如分数、根式、π 等），不要主动化成小数。

【题目】
{question}

【被测AI的回答】
{model_response}

【任务】
从上述回答中提取被测AI给出的最终答案。注意：
1. 只提取最终答案，不是中间计算结果
2. 如果回答里给出了精确表达式与小数近似，优先精确表达式
3. 保留原始表达式形式（可为 LaTeX 或普通表达式）
4. 如果无法提取到有效答案，返回 null

【输出格式】
请只输出以下 JSON，不要有其他内容：
{{"extracted_answer": "<提取到的答案字符串，如无法提取则为null>"}}"""


def parse_judge_result(judge_response: str) -> dict:
    """解析裁判返回的 JSON，容错处理"""
    result = {}
    try:
        result = json.loads(judge_response)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[^}]+\}', judge_response)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = {}
    return {
        "extracted_answer": result.get("extracted_answer"),
        "score_weight": result.get("score_weight"),
        "reason": result.get("reason"),
    }


def build_judge_prompt_with_weight(question: str, correct_answer_text: str, model_response: str, judge_hint: Optional[str] = None) -> str:
    """构建裁判 AI 的 Prompt - 提取最终答案并估计权重"""
    extra = f"\n【额外评分提示】\n{judge_hint}\n" if judge_hint else ""
    return f"""你是一个数学题答案提取与评分专家。请从被测AI的回答中提取最终答案，并基于标准答案给出权重评分。

【题目】
{question}

【标准答案（精确表达式）】
{correct_answer_text}

【被测AI的回答】
{model_response}

【任务】
1. 只提取最终答案，不是中间计算结果
2. 答案尽量保持精确形式（如分数、根式、π 等），不要主动化成小数
3. 如果回答里给出了精确表达式与小数近似，优先精确表达式
4. 只有精确表达式与标准答案等价时才能给满分（score_weight=1）
5. 如果只给数值近似或部分正确，给 (0,1) 的权重
6. 明显不对给 0
7. 若无法提取到有效答案，extracted_answer 为 null，score_weight 为 0
{extra}

【输出格式】
请只输出以下 JSON，不要有其他内容：
{{"extracted_answer": "<提取到的答案字符串，如无法提取则为null>", "score_weight": <0到1的小数>, "reason": "<一句话理由>"}}
"""


def is_exact_required(question: dict) -> bool:
    return question.get("exact_required", True) is True


def get_answer_text(question: dict) -> str:
    if question.get("answer_text"):
        return str(question["answer_text"])
    answer = question.get("answer")
    if isinstance(answer, (int,)) or (isinstance(answer, float) and float(answer).is_integer()):
        return str(int(answer))
    return str(answer)


def coerce_weight(value) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    if weight < 0:
        return 0.0
    if weight > 1:
        return 1.0
    return weight


def normalize_answer_text(text: str) -> str:
    """规范化答案文本，便于字符串精确比对"""
    s = text.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2]
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2]
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s)
    return s


def check_answer(extracted, question: dict) -> bool:
    """检查答案是否正确（数值或精确表达式）"""
    if extracted is None:
        return False
    extracted_text = str(extracted).strip()

    if is_exact_required(question):
        correct_text = get_answer_text(question)
        return normalize_answer_text(extracted_text) == normalize_answer_text(correct_text)

    tolerance = question.get("tolerance", 0)
    correct = question.get("answer")
    try:
        return abs(float(extracted) - float(correct)) <= tolerance
    except:
        return False


def select_questions(req: EvaluateRequest) -> list[dict]:
    data = load_questions()
    questions_map = {q["id"]: q for q in data["questions"]}
    selected = []
    for qid in req.question_ids:
        if qid in questions_map:
            selected.append(questions_map[qid])
    if not selected:
        raise HTTPException(status_code=400, detail="No valid questions selected")
    return selected


def init_progress(eval_id: str, req: EvaluateRequest, selected_questions: list[dict]):
    total_tasks = len(req.tested_models) * len(selected_questions)
    question_status = {
        q["id"]: {"total": len(req.tested_models), "running_count": 0, "completed_count": 0}
        for q in selected_questions
    }
    EVAL_PROGRESS[eval_id] = {
        "status": "running",
        "total_tasks": total_tasks,
        "completed_tasks": 0,
        "current_model": None,
        "current_question_id": None,
        "active_tasks": 0,
        "active_model_streams": 0,
        "active_judge_streams": 0,
        "stream_stage": None,
        "model_stream_chars": 0,
        "model_stream_est_tokens": 0,
        "judge_stream_chars": 0,
        "judge_stream_est_tokens": 0,
        "model_prompt_tokens": 0,
        "model_completion_tokens": 0,
        "model_total_tokens": 0,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
        "model_last_usage": None,
        "judge_last_usage": None,
        "question_status": question_status,
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def update_progress(eval_id: str, **kwargs):
    if eval_id in EVAL_PROGRESS:
        EVAL_PROGRESS[eval_id].update(kwargs)
        EVAL_PROGRESS[eval_id]["updated_at"] = datetime.now().isoformat()


def refresh_stream_stage(progress: dict):
    if progress.get("active_model_streams", 0) > 0 and progress.get("active_judge_streams", 0) > 0:
        progress["stream_stage"] = "both"
    elif progress.get("active_model_streams", 0) > 0:
        progress["stream_stage"] = "model"
    elif progress.get("active_judge_streams", 0) > 0:
        progress["stream_stage"] = "judge"
    else:
        progress["stream_stage"] = None


def parse_usage_counts(usage: Optional[dict]) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("promptTokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("completionTokens") or 0
    total = usage.get("total_tokens") or usage.get("totalTokens") or (prompt + completion)
    try:
        return int(prompt), int(completion), int(total)
    except (TypeError, ValueError):
        return 0, 0, 0


def bump_usage(progress: dict, usage: Optional[dict], prefix: str):
    prompt, completion, total = parse_usage_counts(usage)
    if prompt == 0 and completion == 0 and total == 0:
        return
    progress[f"{prefix}_prompt_tokens"] += prompt
    progress[f"{prefix}_completion_tokens"] += completion
    progress[f"{prefix}_total_tokens"] += total
    progress[f"{prefix}_last_usage"] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def estimate_tokens_from_text(text: str) -> int:
    length = len(text)
    if length <= 0:
        return 0
    # 粗略估计：英文约 4 字符 1 token，中文更密；此处仅用于实时显示
    return max(1, length // 4)


async def execute_evaluation(eval_id: str, req: EvaluateRequest, selected_questions: list[dict], track_progress: bool) -> dict:
    if track_progress:
        init_progress(eval_id, req, selected_questions)

    results = {}
    total_tasks = len(req.tested_models) * len(selected_questions)
    completed = 0
    completed_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    model_locks = {m: asyncio.Lock() for m in req.tested_models}

    for model in req.tested_models:
        results[model] = {
            "total_score": 0,
            "max_score": sum(q["score"] for q in selected_questions),
            "score_rate": 0,
            "details": [],
        }

    async def run_one(model: str, q: dict):
        nonlocal completed
        model_stream_active = False
        judge_stream_active = False
        async with sem:
            if track_progress and eval_id in EVAL_PROGRESS:
                async with progress_lock:
                    progress = EVAL_PROGRESS[eval_id]
                    progress["active_tasks"] += 1
                    qs = progress.get("question_status", {}).get(q["id"])
                    if qs is not None:
                        qs["running_count"] += 1
                    update_progress(
                        eval_id,
                        current_model=model,
                        current_question_id=q["id"],
                        completed_tasks=completed,
                        active_tasks=progress["active_tasks"],
                    )
            tolerance = q.get("tolerance", 0)

            try:
                test_messages = [
                    {"role": "user", "content": f"请解答以下数学题，给出详细的解题过程和最终答案：\n\n{q['question']}"}
                ]

                if track_progress and eval_id in EVAL_PROGRESS:
                    progress = EVAL_PROGRESS[eval_id]
                    progress["active_model_streams"] += 1
                    model_stream_active = True
                    refresh_stream_stage(progress)
                    update_progress(eval_id)

                async def on_model_chunk(delta: str):
                    if not track_progress or eval_id not in EVAL_PROGRESS:
                        return
                    progress = EVAL_PROGRESS[eval_id]
                    progress["model_stream_chars"] += len(delta)
                    progress["model_stream_est_tokens"] += estimate_tokens_from_text(delta)
                    update_progress(eval_id)

                model_response, model_usage = await call_openrouter_stream(
                    OPENROUTER_API_KEY,
                    model,
                    test_messages,
                    drop_params=["temperature", "max_tokens"],
                    on_chunk=on_model_chunk if track_progress else None,
                )

                if track_progress and eval_id in EVAL_PROGRESS:
                    progress = EVAL_PROGRESS[eval_id]
                    if progress["model_stream_est_tokens"] == 0 and model_response:
                        progress["model_stream_est_tokens"] = estimate_tokens_from_text(model_response)
                        progress["model_stream_chars"] = len(model_response)
                    bump_usage(progress, model_usage, "model")
                    update_progress(eval_id)

                answer_text = get_answer_text(q)
                judge_prompt = build_judge_prompt_with_weight(
                    q["question"],
                    answer_text,
                    model_response,
                    q.get("judge_hint"),
                )
                judge_messages = [{"role": "user", "content": judge_prompt}]

                if track_progress and eval_id in EVAL_PROGRESS:
                    progress = EVAL_PROGRESS[eval_id]
                    progress["active_judge_streams"] += 1
                    judge_stream_active = True
                    refresh_stream_stage(progress)
                    update_progress(eval_id)

                async def on_judge_chunk(delta: str):
                    if not track_progress or eval_id not in EVAL_PROGRESS:
                        return
                    progress = EVAL_PROGRESS[eval_id]
                    progress["judge_stream_chars"] += len(delta)
                    progress["judge_stream_est_tokens"] += estimate_tokens_from_text(delta)
                    update_progress(eval_id)

                judge_response, judge_usage = await call_openrouter_stream(
                    OPENROUTER_API_KEY,
                    req.judge_model,
                    judge_messages,
                    extra_params={"reasoning": {"effort": "high"}} if req.judge_model == "openai/gpt-5.2" else None,
                    drop_params=["temperature"] if req.judge_model == "openai/gpt-5.2" else None,
                    on_chunk=on_judge_chunk if track_progress else None,
                )

                if track_progress and eval_id in EVAL_PROGRESS:
                    progress = EVAL_PROGRESS[eval_id]
                    if progress["judge_stream_est_tokens"] == 0 and judge_response:
                        progress["judge_stream_est_tokens"] = estimate_tokens_from_text(judge_response)
                        progress["judge_stream_chars"] = len(judge_response)
                    bump_usage(progress, judge_usage, "judge")
                    update_progress(eval_id)

                judge_result = parse_judge_result(judge_response)
                extracted_answer = judge_result.get("extracted_answer")
                judge_reason = judge_result.get("reason")
                if extracted_answer is None:
                    extracted_answer = parse_extracted_answer(judge_response)

                weight = 0.0
                if extracted_answer is not None and is_exact_required(q):
                    if normalize_answer_text(str(extracted_answer)) == normalize_answer_text(answer_text):
                        weight = 1.0
                    else:
                        weight = min(coerce_weight(judge_result.get("score_weight")), 0.99)
                elif extracted_answer is not None and not is_exact_required(q):
                    weight = 1.0 if check_answer(extracted_answer, q) else 0.0

                earned_score = q["score"] * weight
                is_correct = weight >= 1.0

                if is_correct:
                    reason = f"答案正确: {extracted_answer}"
                elif extracted_answer is not None:
                    if judge_reason:
                        reason = f"{judge_reason}"
                    else:
                        reason = f"答案错误: 提取到 {extracted_answer}，标准答案 {answer_text}"
                else:
                    reason = "无法从回答中提取有效答案"

                correct_answer_display = answer_text if is_exact_required(q) else q["answer"]
                detail = {
                    "question_id": q["id"],
                    "question_score": q["score"],
                    "model_response": model_response,
                    "model_final_answer": extracted_answer,
                    "correct_answer": correct_answer_display,
                    "tolerance": tolerance,
                    "is_correct": is_correct,
                    "score_ratio": weight,
                    "earned_score": earned_score,
                    "reason": reason,
                    "score_weight": weight,
                    "judge_response": judge_response,
                    "judge_reason": judge_reason,
                    "model_usage": model_usage,
                    "judge_usage": judge_usage,
                }

                update_question_stats(q["id"], is_correct)

                async with model_locks[model]:
                    results[model]["total_score"] += earned_score
                    results[model]["details"].append(detail)

            except Exception as e:
                detail = {
                    "question_id": q["id"],
                    "question_score": q["score"],
                    "model_response": None,
                    "model_final_answer": None,
                    "correct_answer": get_answer_text(q) if is_exact_required(q) else q["answer"],
                    "tolerance": tolerance,
                    "is_correct": False,
                    "score_ratio": 0,
                    "earned_score": 0,
                    "reason": f"错误: {str(e)}",
                    "score_weight": 0,
                    "judge_response": None,
                    "judge_reason": None,
                    "model_usage": None,
                    "judge_usage": None,
                }
                async with model_locks[model]:
                    results[model]["details"].append(detail)
                update_question_stats(q["id"], False)

            finally:
                if track_progress and eval_id in EVAL_PROGRESS:
                    async with progress_lock:
                        progress = EVAL_PROGRESS[eval_id]
                        if model_stream_active:
                            progress["active_model_streams"] = max(0, progress["active_model_streams"] - 1)
                        if judge_stream_active:
                            progress["active_judge_streams"] = max(0, progress["active_judge_streams"] - 1)
                        refresh_stream_stage(progress)
                        progress["active_tasks"] = max(0, progress["active_tasks"] - 1)
                        qs = progress.get("question_status", {}).get(q["id"])
                        if qs is not None:
                            qs["running_count"] = max(0, qs["running_count"] - 1)
                            qs["completed_count"] += 1
                        update_progress(eval_id, active_tasks=progress["active_tasks"])

                async with completed_lock:
                    completed += 1
                    if track_progress:
                        update_progress(eval_id, completed_tasks=completed)

    tasks = [run_one(model, q) for model in req.tested_models for q in selected_questions]
    await asyncio.gather(*tasks)

    for model in req.tested_models:
        if results[model]["max_score"] > 0:
            results[model]["score_rate"] = results[model]["total_score"] / results[model]["max_score"]
            results[model]["normalized_total_score"] = results[model]["score_rate"] * 100
        record_model_eval(
            eval_id,
            model,
            results[model]["total_score"],
            results[model]["max_score"],
            results[model]["score_rate"],
            results[model]["normalized_total_score"],
        )

    eval_result = {
        "id": eval_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "tested_models": req.tested_models,
            "judge_model": req.judge_model,
            "question_count": len(selected_questions),
        },
        "results": results,
    }

    result_file = RESULTS_DIR / f"{eval_id}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    if track_progress:
        update_progress(
            eval_id,
            status="completed",
            completed_tasks=total_tasks,
        )

    return eval_result


# ==================== API 路由 ====================

# 前端页面
@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "index.html")


# ---------- 题库管理 ----------

@app.get("/api/questions")
async def get_questions():
    """获取所有题目"""
    data = load_questions()
    stats = get_question_stats_map()
    for q in data.get("questions", []):
        st = stats.get(q["id"], {"correct_count": 0, "attempt_count": 0})
        correct = st["correct_count"]
        attempt = st["attempt_count"]
        q["correct_count"] = correct
        q["attempt_count"] = attempt
        q["correct_rate"] = (correct / attempt) if attempt > 0 else None
    return data


@app.post("/api/questions")
async def create_question(q: QuestionCreate):
    """新增题目"""
    data = load_questions()
    new_question = {
        "id": generate_question_id(),
        "question": q.question,
        "answer": q.answer,
        "score": q.score,
        "tolerance": q.tolerance or 0,
    }
    if q.answer_text:
        new_question["answer_text"] = q.answer_text
    if q.exact_required:
        new_question["exact_required"] = True
    if q.judge_hint:
        new_question["judge_hint"] = q.judge_hint
    data["questions"].append(new_question)
    save_questions(data)
    return new_question


@app.put("/api/questions/{question_id}")
async def update_question(question_id: str, q: QuestionCreate):
    """修改题目"""
    data = load_questions()
    for i, question in enumerate(data["questions"]):
        if question["id"] == question_id:
            updated_question = {
                "id": question_id,
                "question": q.question,
                "answer": q.answer,
                "score": q.score,
                "tolerance": q.tolerance or 0,
            }
            if q.answer_text:
                updated_question["answer_text"] = q.answer_text
            if q.exact_required:
                updated_question["exact_required"] = True
            if q.judge_hint:
                updated_question["judge_hint"] = q.judge_hint
            data["questions"][i] = updated_question
            save_questions(data)
            return data["questions"][i]
    raise HTTPException(status_code=404, detail="Question not found")


@app.delete("/api/questions/{question_id}")
async def delete_question(question_id: str):
    """删除题目"""
    data = load_questions()
    original_len = len(data["questions"])
    data["questions"] = [q for q in data["questions"] if q["id"] != question_id]
    if len(data["questions"]) == original_len:
        raise HTTPException(status_code=404, detail="Question not found")
    save_questions(data)
    return {"message": "Deleted"}


# ---------- 模型列表 ----------

@app.get("/api/models")
async def get_models():
    """获取 OpenRouter 可用模型列表"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=headers,
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch models")

        result = response.json()
        # 返回模型ID列表
        models = [m["id"] for m in result.get("data", [])]
        return {"models": sorted(models)}


# ---------- 测评执行 ----------

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    """执行测评"""
    selected_questions = select_questions(req)
    eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return await execute_evaluation(eval_id, req, selected_questions, track_progress=False)


@app.post("/api/evaluate/start")
async def evaluate_start(req: EvaluateRequest):
    """启动测评（异步进度）"""
    selected_questions = select_questions(req)
    eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    asyncio.create_task(execute_evaluation(eval_id, req, selected_questions, track_progress=True))
    return {"id": eval_id}


@app.get("/api/evaluate/progress/{eval_id}")
async def evaluate_progress(eval_id: str):
    """获取测评进度"""
    if eval_id not in EVAL_PROGRESS:
        raise HTTPException(status_code=404, detail="Eval not found")
    return EVAL_PROGRESS[eval_id]


@app.get("/api/evaluate/running")
async def evaluate_running():
    """获取正在进行中的测评"""
    running = [
        {"id": eval_id, **info}
        for eval_id, info in EVAL_PROGRESS.items()
        if info.get("status") == "running"
    ]
    return {"running": running}


@app.get("/api/openrouter/health")
async def openrouter_health():
    """OpenRouter 连接状态"""
    now_ts = datetime.now().timestamp()
    if OPENROUTER_HEALTH["ok"] is not None and (now_ts - OPENROUTER_HEALTH["checked_at"]) < OPENROUTER_HEALTH_TTL:
        return OPENROUTER_HEALTH
    try:
        start = datetime.now().timestamp()
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers)
        latency = int((datetime.now().timestamp() - start) * 1000)
        ok = resp.status_code == 200
        OPENROUTER_HEALTH.update({"ok": ok, "checked_at": now_ts, "latency_ms": latency})
        return OPENROUTER_HEALTH
    except Exception:
        OPENROUTER_HEALTH.update({"ok": False, "checked_at": now_ts, "latency_ms": None})
        return OPENROUTER_HEALTH


@app.get("/api/ui/state")
async def get_ui_state():
    """获取前端状态"""
    return get_ui_state_db()


@app.post("/api/ui/state")
async def set_ui_state(state: UiState):
    """保存前端状态"""
    return set_ui_state_db(state)


# ---------- 结果查询 ----------

@app.get("/api/results")
async def get_results():
    """获取所有历史测评结果"""
    results = []
    for file in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 只返回摘要信息
            results.append({
                "id": data["id"],
                "timestamp": data["timestamp"],
                "config": data["config"],
                "summary": {
                    model: {
                        "score_rate": info["score_rate"],
                        "total_score": info["total_score"],
                        "max_score": info["max_score"],
                        "normalized_total_score": info.get("normalized_total_score"),
                    }
                    for model, info in data["results"].items()
                }
            })
    return {"results": results}


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """获取单次测评详情"""
    result_file = RESULTS_DIR / f"{result_id}.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    with open(result_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
