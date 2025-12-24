# AI 数学 Benchmark 系统

本项目用于测评 AI 数学能力：前端展示题库与测评进度，后端调用 OpenRouter 模型并由裁判模型给分。

## ⚠️ 安全与发布约束（必须遵守）

- **禁止将任何 API Key 上传或公开**（包括截图、日志、代码片段）。
- 上传到 GitHub 前，务必确认 **仓库中不含密钥**，只通过 `.env` 或环境变量提供。
- 如需公开发布，请先彻底检查历史提交中是否存在密钥。

## 快速启动

方式一：双击 `start.command`

方式二：命令行
```bash
cd 数学benchmark
export OPENROUTER_API_KEY="你的key"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

浏览器访问：
```
http://localhost:8000
```

## 依赖

```bash
python -m pip install -r requirements.txt
```

## 配置 OpenRouter API Key（本地）

必须通过环境变量提供，不写入仓库：

```bash
export OPENROUTER_API_KEY="你的key"
```

## 目录结构

```
数学benchmark/
├── index.html              # 前端页面（KaTeX 渲染）
├── server.py               # 后端服务（包含 OpenRouter Key）
├── requirements.txt        # 依赖
├── data/
│   ├── questions.json      # 题库
│   └── stats.db            # SQLite 统计数据库
└── results/
    └── eval_*.json         # 测评结果存档
```

## 核心说明

- 裁判模型用于抽取答案并按规则打分。
- 题目评分统一为 5 分制，总分 100。
- 评测结果与正确率统计持久化在 SQLite 中。

## 注意事项

- 请勿在任何公共平台分享 `server.py` 中的 API Key。
- 如需更换密钥，仅在本地修改并妥善保管。
