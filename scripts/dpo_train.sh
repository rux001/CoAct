#!/bin/bash
# Unified DPO training script for GSM8K / MATH / WebInstruct.
# Usage: bash dpo_train.sh --dataset <gsm8k|math|webinstruct> --iteration <N> [--output_root <dir>]
#
# Required env vars (from run_iteration.sh): HF_TOKEN, LLAMAFACTORY_DIR
# Optional env vars: BASE_MODEL, TEMPLATE, MODEL_SIZE, NUM_GPUS

set -euo pipefail

DATASET=""
ITERATION=""
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)      DATASET="$2"; shift 2 ;;
        --iteration)    ITERATION="$2"; shift 2 ;;
        --output_root)  OUTPUT_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ] || [ -z "$ITERATION" ]; then
    echo "Usage: $0 --dataset <gsm8k|math|webinstruct> --iteration <N> [--output_root <dir>]"
    exit 1
fi

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AL_DIR="$( cd "${SCRIPTS_DIR}/.." && pwd )"
CONFIGS_DIR="${AL_DIR}/configs"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AL_DIR}/output}"

# Per-dataset base model, template, model size (overridable via env)
case "$DATASET" in
    gsm8k)
        BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_GSM8K:-meta-llama/Meta-Llama-3-8B}}"
        TEMPLATE="${TEMPLATE:-alpaca}"
        MODEL_SIZE="${MODEL_SIZE:-llama3-8b}"
        EVAL_DATASET="gsm8k_llama3_8b_preference_pairs"
        ;;
    math)
        BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_MATH:-meta-llama/Meta-Llama-3-8B}}"
        TEMPLATE="${TEMPLATE:-alpaca}"
        MODEL_SIZE="${MODEL_SIZE:-llama3-8b}"
        EVAL_DATASET="math_llama3_8b_preference_pairs"
        ;;
    webinstruct)
        BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_WEBINSTRUCT:-meta-llama/Meta-Llama-3-8B-Instruct}}"
        TEMPLATE="${TEMPLATE:-llama3}"
        MODEL_SIZE="${MODEL_SIZE:-llama3-8b-instruct}"
        EVAL_DATASET="webinstruct_llama3_8b_preference_pairs"
        ;;
    *) echo "Error: unsupported dataset '$DATASET'"; exit 1 ;;
esac

DATASET_NAME="${DATASET}_AL_it${ITERATION}_preference_pairs"
VAL_DATASET="${LLAMAFACTORY_DIR}/data/${DATASET}_AL_dev.json"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/it${ITERATION}/model"
NUM_GPUS="${NUM_GPUS:-4}"

echo "DPO Training: dataset=$DATASET iteration=$ITERATION model=$BASE_MODEL output=$OUTPUT_DIR gpus=$NUM_GPUS"

mkdir -p "$OUTPUT_DIR"

# Dataset validation
if [ ! -f "${LLAMAFACTORY_DIR}/data/${DATASET_NAME}.json" ]; then
    echo "Error: preference dataset not found at ${LLAMAFACTORY_DIR}/data/${DATASET_NAME}.json"
    exit 1
fi
if [ ! -f "$VAL_DATASET" ]; then
    echo "Error: validation dataset not found at $VAL_DATASET"
    exit 1
fi

# Determine adapter path from previous iteration
TRAIN_MODEL_PATH="$BASE_MODEL"
TRAIN_ADAPTER_PATH=""
if [ "$ITERATION" -ne 0 ]; then
    PREV_IT=$((ITERATION-1))
    PREV_MODEL_DIR="${OUTPUT_ROOT}/${DATASET}/it${PREV_IT}/model"
    if [ -d "$PREV_MODEL_DIR" ]; then
        PREV_LATEST_CKPT=$(ls -td "${PREV_MODEL_DIR}"/checkpoint-* 2>/dev/null | head -n 1 || true)
        if [ -n "$PREV_LATEST_CKPT" ]; then
            TRAIN_ADAPTER_PATH="${PREV_LATEST_CKPT//:/}"
        else
            TRAIN_ADAPTER_PATH="$PREV_MODEL_DIR"
        fi
        echo "Resuming from adapter: $TRAIN_ADAPTER_PATH"
    else
        echo "No previous model dir ($PREV_MODEL_DIR); training from base."
    fi
fi

# Generate accelerate + DPO configs
TMP_CFG_DIR="${LLAMAFACTORY_DIR}/configs/al_camera_ready_${DATASET}_it${ITERATION}"
mkdir -p "$TMP_CFG_DIR"

cat > "${TMP_CFG_DIR}/accelerate_config.yaml" << ACCEOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_clipping: 'auto'
  offload_optimizer_device: none
  offload_param_device: none
  zero_stage: 2
machine_rank: 0
main_process_port: 29501
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
rdzv_backend: static
same_network: true
use_cpu: false
ACCEOF

export TRAIN_MODEL_PATH DATASET_NAME VAL_DATASET OUTPUT_DIR TEMPLATE EVAL_DATASET
envsubst '${TRAIN_MODEL_PATH} ${DATASET_NAME} ${VAL_DATASET} ${OUTPUT_DIR} ${TEMPLATE} ${EVAL_DATASET}' \
    < "${CONFIGS_DIR}/dpo_${DATASET}.yaml" > "${TMP_CFG_DIR}/dpo_config.yaml"

if [ -n "$TRAIN_ADAPTER_PATH" ]; then
    echo "adapter_name_or_path: ${TRAIN_ADAPTER_PATH}" >> "${TMP_CFG_DIR}/dpo_config.yaml"
fi

# Launch training
cd "$LLAMAFACTORY_DIR"
accelerate launch \
    --config_file "${TMP_CFG_DIR}/accelerate_config.yaml" \
    src/train.py \
    "${TMP_CFG_DIR}/dpo_config.yaml"

echo "DPO training complete. Output: $OUTPUT_DIR"
