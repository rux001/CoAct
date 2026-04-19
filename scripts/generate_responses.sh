#!/bin/bash
# Generate K=8 responses per question at different temperatures via vLLM.
set -euo pipefail

BASE_MODEL=""
ADAPTER_PATH=""
DATASET=""
RESULTS_DIR=""
TEMPLATE="alpaca"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       BASE_MODEL="$2"; shift 2 ;;
        --adapter)     ADAPTER_PATH="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --template)    TEMPLATE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$BASE_MODEL" ] || [ -z "$DATASET" ] || [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 --model <model> --dataset <dataset> --results_dir <dir> [--adapter <path>] [--template <template>]"
    exit 1
fi

LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR to your LLaMA-Factory checkout}"
mkdir -p "$RESULTS_DIR"
cd "$LLAMAFACTORY_DIR"

temperatures=(0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7)

for i in {1..8}; do
    temp="${temperatures[$((i-1))]}"
    seed=$(date +%s%N | cut -b1-13)
    echo "Run $i/8: temperature=$temp seed=$seed"

    VLLM_CMD="CUDA_VISIBLE_DEVICES=0 python scripts/vllm_infer.py \
        --model_name_or_path \"$BASE_MODEL\" \
        --dataset \"$DATASET\" \
        --template \"$TEMPLATE\" \
        --save_name \"${RESULTS_DIR}/run${i}_t${temp}.jsonl\" \
        --temperature $temp \
        --top_p 0.9 \
        --top_k 40 \
        --max_new_tokens 2048 \
        --repetition_penalty 1.1 \
        --enable_thinking False \
        --seed \"$seed\""

    if [ -n "$ADAPTER_PATH" ]; then
        VLLM_CMD="$VLLM_CMD --adapter_name_or_path \"$ADAPTER_PATH\""
    fi

    eval $VLLM_CMD
done

echo "All 8 runs completed. Output: $RESULTS_DIR/"
