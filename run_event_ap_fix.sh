#!/usr/bin/env bash
#
# run_event_ap_fix.sh — 运行 event_ap_fix.py（A&P 生成）
#
# 用法:
#   ./run_event_ap_fix.sh                    # 使用默认参数 (mistral, gt)
#   ./run_event_ap_fix.sh mistral gt         # 指定模型和设置
#   ./run_event_ap_fix.sh qwen gen           # 使用 qwen 模型 + gen 设置
#
# 可用模型: mistral, qwen, deepseek, llama3, llama2
# 可用设置: gt (真实笔记), gen (生成笔记)
#
# 显卡: 第7块
# Conda 环境: safevla

set -euo pipefail

# ===== 配置 =====
MODEL="${1:-mistral}"
SETTING="${2:-gt}"
GPU_DEVICES="7"
HF_TOKEN=""

# 脚本所在目录: long_data_related/longitudinal_clinical_summarization/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 项目根目录: dongshuwei/
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/modeling/event_ap_fix.py"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/event_ap_fix_${MODEL}_${SETTING}.log"

# ===== 检查 =====
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到脚本 $SCRIPT_PATH"
    exit 1
fi

# ===== 创建日志目录 =====
mkdir -p "$LOG_DIR"

# ===== 运行 =====
echo "========================================"
echo "  event_ap_fix.py 启动"
echo "  模型:     $MODEL"
echo "  设置:     $SETTING"
echo "  显卡:     $GPU_DEVICES"
echo "  日志:     $LOG_FILE"
echo "  Conda:    safevla"
echo "========================================"

# 使用 bash -c 激活 conda 环境，避免 conda run 的输出缓冲问题
nohup bash -c "
  export HF_TOKEN='$HF_TOKEN'
  export CUDA_VISIBLE_DEVICES='$GPU_DEVICES'
  export PYTHONUNBUFFERED=1
  source /home/csuvla/miniconda3/etc/profile.d/conda.sh
  conda activate safevla
  cd '$PROJECT_DIR'
  python -u '$SCRIPT_PATH' \
    --inputdir data/AP/input \
    --outputdir data/AP/generated \
    --setting '$SETTING' \
    --model '$MODEL'
" > "$LOG_FILE" 2>&1 &

PID=$!
echo "进程 PID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "查看实时日志: tail -f $LOG_FILE"
echo "查看 GPU 状态: watch -n 2 nvidia-smi"
echo "检查进程: ps aux | grep event_ap_fix"
