#!/bin/bash

# AI 数学 Benchmark 启动脚本
# 双击此文件即可启动服务

cd "$(dirname "$0")"

echo "================================"
echo "  AI 数学 Benchmark 系统"
echo "================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python"
    echo "下载地址: https://www.python.org/downloads/"
    read -p "按回车键退出..."
    exit 1
fi

echo "Python 版本: $(python3 --version)"
echo ""

# 加载本地 .env（不会纳入仓库）
if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
fi

# 检查 API Key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "错误: 缺少 OPENROUTER_API_KEY，请在本地环境变量或 .env 中设置。"
    echo "示例: OPENROUTER_API_KEY=sk-or-***"
    read -p "按回车键退出..."
    exit 1
fi

# 检查并安装依赖
echo "检查依赖..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
    echo ""
fi

# 启动服务
echo "启动服务..."
echo "服务地址: http://localhost:8000"
echo ""
echo "按 Ctrl+C 停止服务"
echo "================================"
echo ""

# 2秒后打开浏览器
(sleep 2 && open "http://localhost:8000") &

# 启动服务器
python3 server.py
