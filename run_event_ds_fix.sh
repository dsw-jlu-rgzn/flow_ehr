#!/usr/bin/env bash
#
# run_event_ds_fix.sh — 运行 event_ds_fix.py（DS 出院总结生成）
#
# 用法:
#   ./run_event_ds_fix.sh                    # 使用默认参数 (mistral)
#   ./run_event_ds_fix.sh qwen               # 使用 qwen 模型
#
# 可用模型: mistral, qwen, deepseek, llama3, llama2
#
# 显卡: 第6,7块（代码硬编码 CUDA_VISIBLE_DEVICES='6,7'）
# Conda 环境: safevla

set -euo pipefail

# ===== 配置 =====
MODEL="${1:-mistral}"
GPU_DEVICES="6,7"
HF_TOKEN=""

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 项目根目录: /home/csuvla/dongshuwei/
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/modeling/event_ds_fix.py"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/event_ds_fix_${MODEL}.log"

# ===== 检查 =====
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到脚本 $SCRIPT_PATH"
    exit 1
fi

# ===== 创建日志目录 =====
mkdir -p "$LOG_DIR"

# ===== 打印配置 =====
echo "========================================"
echo "  event_ds_fix.py 启动"
echo "  模型:     $MODEL"
echo "  显卡:     $GPU_DEVICES"
echo "  输入:     data/DS/full_input"
echo "  输出:     data/DS/generated/EE/$MODEL/"
echo "  日志:     $LOG_FILE"
echo "  Conda:    safevla"
echo "========================================"

# 使用 bash -c 激活 conda 环境，避免 conda run 的输出缓冲问题
nohup bash -c "
  export HF_TOKEN='$HF_TOKEN'
  export PYTHONUNBUFFERED=1
  source /home/csuvla/miniconda3/etc/profile.d/conda.sh
  conda activate safevla
  cd '$PROJECT_DIR'
  python -u '$SCRIPT_PATH' \
    --inputdir data/DS/full_input \
    --outputdir data/DS/generated \
    --model '$MODEL'
" > "$LOG_FILE" 2>&1 &

PID=$!
echo ""
echo "进程 PID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "查看实时日志: tail -f $LOG_FILE"
echo "查看 GPU 状态: watch -n 2 nvidia-smi"
echo "检查进程: ps aux | grep event_ds_fix"
