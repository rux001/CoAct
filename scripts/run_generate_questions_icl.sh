#!/bin/bash
# ICL question generation pipeline with self-consistency responses.
# Usage: bash run_generate_questions_icl.sh <iteration> <oracle_budget> <num_prompts> <model_path> <adapter_path> <dataset> [output_root]

set -euo pipefail

ITERATION=${1:-0}
ORACLE_BUDGET=${2:-150}
NUM_PROMPTS=${3:-1000}
MODEL_PATH=${4:-"meta-llama/Meta-Llama-3-8B-Instruct"}
ADAPTER_PATH=${5:-""}
DATASET=${6:-"gsm8k"}
DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
NUM_FEW_SHOT=3

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AL_DIR="$( cd "${SCRIPTS_DIR}/.." && pwd )"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR}"
OUTPUT_ROOT="${7:-${AL_DIR}/output}"
IT_DIR="${OUTPUT_ROOT}/${DATASET}/it${ITERATION}"
ICL_DIR="${IT_DIR}/icl"
mkdir -p "$ICL_DIR"

ORACLE_FILE="${IT_DIR}/oracle/eval_high_conf.json"
PROMPTS_FILE="${ICL_DIR}/prompts.json"
QUESTIONS_FILE="${ICL_DIR}/questions.jsonl"
DATASET_FILE="${ICL_DIR}/dataset.json"

echo "ICL Pipeline: iteration=$ITERATION budget=$ORACLE_BUDGET prompts=$NUM_PROMPTS dataset=$DATASET"

# Step 1: Prepare ICL prompts from oracle-verified questions
python3 "${SCRIPTS_DIR}/generate_new_questions_icl.py" \
    --oracle_file "$ORACLE_FILE" \
    --output_file "$PROMPTS_FILE" \
    --num_prompts $NUM_PROMPTS \
    --num_few_shot $NUM_FEW_SHOT \
    --dataset_type $DATASET \
    --seed 42

# Register prompts in LLaMA-Factory data dir
PROMPTS_LF_NAME="icl_prompts_it${ITERATION}_${DATASET}"
ln -sf "$PROMPTS_FILE" "${LLAMAFACTORY_DIR}/data/${PROMPTS_LF_NAME}.json"

# Step 2: Generate new questions
seed=$(date +%s%N | cut -b1-13)
cd "$LLAMAFACTORY_DIR"

VLLM_CMD="CUDA_VISIBLE_DEVICES=0 python scripts/vllm_infer.py \
    --model_name_or_path \"$MODEL_PATH\" \
    --dataset \"${PROMPTS_LF_NAME}\" \
    --template \"${TEMPLATE:-alpaca}\" \
    --save_name \"${QUESTIONS_FILE}\" \
    --temperature 0.5 \
    --top_p 0.9 \
    --top_k 40 \
    --max_new_tokens 1024 \
    --repetition_penalty 1.1 \
    --enable_thinking False \
    --seed $seed"

if [ -n "$ADAPTER_PATH" ]; then
    VLLM_CMD="$VLLM_CMD --adapter_name_or_path \"$ADAPTER_PATH\""
fi
eval $VLLM_CMD

# Step 3: Convert generated questions to response-generation format
DATASET_LF_NAME="icl_generated_it${ITERATION}_${DATASET}"
python3 "${SCRIPTS_DIR}/generate_new_responses_icl.py" \
    --input_file "$QUESTIONS_FILE" \
    --output_file "$DATASET_FILE" \
    --dataset_type "$DATASET"

ln -sf "$DATASET_FILE" "${LLAMAFACTORY_DIR}/data/${DATASET_LF_NAME}.json"

# Step 4: Generate 8 self-consistency responses
temperatures=(0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70)

for i in {1..8}; do
    temp="${temperatures[$((i-1))]}"
    seed=$(date +%s%N | cut -b1-13)
    RUN_OUTPUT="${ICL_DIR}/run${i}_t${temp}.jsonl"

    VLLM_CMD="CUDA_VISIBLE_DEVICES=0 python scripts/vllm_infer.py \
        --model_name_or_path \"$MODEL_PATH\" \
        --dataset \"${DATASET_LF_NAME}\" \
        --template ${TEMPLATE:-alpaca} \
        --save_name \"${RUN_OUTPUT}\" \
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
    echo "Run $i/8 (temp=$temp) complete"
done

echo "ICL pipeline complete. Outputs in: $ICL_DIR"
