#!/bin/bash
# .devcontainer/postCreateCommand.sh
# RAGSDK Dev Container 初始化脚本

set -e

echo "=========================================="
echo "RAGSDK Dev Container post-create setup..."
echo "=========================================="

# 1. 验证 NPU 环境
if command -v npu-smi &> /dev/null; then
    echo "NPU environment detected:"
    npu-smi info || true
else
    echo "Warning: npu-smi not found. NPU may not be available."
fi

# 2. 验证 IndexSDK
if [ -d "/usr/local/Ascend/mxIndex" ]; then
    echo "IndexSDK detected at /usr/local/Ascend/mxIndex"
else
    echo "Warning: IndexSDK not found at /usr/local/Ascend/mxIndex"
fi

# 3. 验证 Faiss
if [ -d "/usr/local/faiss/faiss1.10.0" ]; then
    echo "Faiss detected at /usr/local/faiss/faiss1.10.0"
else
    echo "Warning: Faiss not found at /usr/local/faiss/faiss1.10.0"
fi

# 4. 验证 ascendfaiss
python3 -c "import ascendfaiss; print('ascendfaiss version:', ascendfaiss.__version__)" 2>/dev/null || \
    echo "Warning: ascendfaiss not importable"

# 5. source Ascend Toolkit 环境
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "Ascend Toolkit environment sourced."
fi

echo "=========================================="
echo "RAGSDK Dev Container setup completed!"
echo ""
echo "Quick start:"
echo "  pip install -e .               # 以可编辑模式安装 RAGSDK"
echo "  pytest tests/                  # 运行测试"
echo "=========================================="
