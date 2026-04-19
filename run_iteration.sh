#!/bin/bash
# Selective DPO Pipeline — complete iteration runner.
# Usage: ./run_iteration.sh <iteration> [--dataset gsm8k|math|webinstruct] [--skip-generation] [--skip-icl] [--skip-oracle] [--skip-dpo]

set -euo pipefail

ITERATION=${1:-}
if [ -z "$ITERATION" ]; then
    echo "Usage: $0 <iteration> [--dataset <gsm8k|math|webinstruct>] [--skip-generation] [--skip-icl] [--skip-oracle] [--skip-dpo]"
    exit 1
fi

SKIP_GENERATION=false
SKIP_ICL=false
SKIP_ORACLE=false
SKIP_DPO=false
DATASET="gsm8k"

shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-generation) SKIP_GENERATION=true; shift ;;
        --skip-icl)        SKIP_ICL=true; shift ;;
        --skip-oracle)     SKIP_ORACLE=true; shift ;;
        --skip-dpo)        SKIP_DPO=true; shift ;;
        --dataset)         DATASET="$2"; shift 2 ;;
        --dataset=*)       DATASET="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
case "$DATASET" in
    gsm8k|math|webinstruct) ;;
    *) echo "Error: unsupported dataset '$DATASET'"; exit 1 ;;
esac

# Environment
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HUGGINGFACE_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=1
export DEEPSPEED_DISABLE_AUTOTUNE=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# Paths
AL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPTS_DIR="${AL_DIR}/scripts"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TEMPLATE="${TEMPLATE:-llama3}"
export TEMPLATE
MODEL_SIZE="${MODEL_SIZE:-llama3-8b-instruct}"
OUTPUT_ROOT="${AL_DIR}/output"
IT_DIR="${OUTPUT_ROOT}/${DATASET}/it${ITERATION}"

case "$DATASET" in
    gsm8k)       CONSISTENCY_THRESHOLD=5 ;;
    math)        CONSISTENCY_THRESHOLD=5 ;;
    webinstruct) CONSISTENCY_THRESHOLD=7 ;;
esac

ORACLE_BUDGET=150
NUM_PROMPTS=2000
PREV_ITERATION=$((ITERATION > 0 ? ITERATION - 1 : 0))
PREV_IT_DIR="${OUTPUT_ROOT}/${DATASET}/it${PREV_ITERATION}"

# Create output directories
mkdir -p "${IT_DIR}"/{responses,consistency,oracle,icl,training,model}
LOG_DIR="${AL_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${DATASET}_it${ITERATION}_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

run_step() {
    local name="$1" cmd="$2"
    log "Starting Step: $name"
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        log "Step completed successfully: $name"
    else
        log "Step failed: $name"
        exit 1
    fi
}

clear_gpu() {
    python -c "
import torch, gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    gc.collect()
" 2>/dev/null || true
}

log "Pipeline: dataset=$DATASET iteration=$ITERATION model=$BASE_MODEL template=$TEMPLATE"

# --- Step 1: Generate K=8 Responses ---
if [ "$SKIP_GENERATION" = false ]; then
    clear_gpu

    # For generation, need the LF dataset registered
    GENERATION_DATASET="${DATASET}_al_it${ITERATION}"
    PREV_ADAPTER=""
    if [ "$ITERATION" -gt 0 ]; then
        PREV_CKPT=$(ls -td "${PREV_IT_DIR}/model"/checkpoint-* 2>/dev/null | head -n 1 | tr -d ':' || true)
        if [ -n "$PREV_CKPT" ]; then
            PREV_ADAPTER="$PREV_CKPT"
        elif [ -d "${PREV_IT_DIR}/model" ]; then
            PREV_ADAPTER="${PREV_IT_DIR}/model"
        fi
    fi

    run_step "Generate K Responses" "
        bash '${SCRIPTS_DIR}/generate_responses.sh' \
            --model '$BASE_MODEL' \
            --adapter '$PREV_ADAPTER' \
            --dataset '$GENERATION_DATASET' \
            --results_dir '${IT_DIR}/responses' \
            --template '$TEMPLATE'
    "
else
    log "Skipping generation"
fi

# --- Step 2: Convert to K Responses ---
run_step "Convert to K Responses" "
    python '${SCRIPTS_DIR}/make_k_responses.py' \
        --input_dir '${IT_DIR}/responses' \
        --output '${IT_DIR}/responses/k_responses.json'
"

# --- Step 3: Self-Consistency Analysis ---
KNN_FLAG=""
[ "$ITERATION" -gt 0 ] && KNN_FLAG="--use_knn_selection"

run_step "Self-Consistency Analysis" "
    cd '${AL_DIR}' &&
    python '${SCRIPTS_DIR}/self_consistency.py' \
        --dataset it$ITERATION \
        --dataset-prefix '$DATASET' \
        --k_responses_file '${IT_DIR}/responses/k_responses.json' \
        --jsonl_files_pattern '${IT_DIR}/responses/run*_t*.jsonl' \
        --output_root '$OUTPUT_ROOT' \
        --iteration $ITERATION \
        --high_consistency_threshold 4 \
        --n_high_consistency 150 \
        --n_oracle 150 \
        $KNN_FLAG \
        --llama_factory_output_dir '${LLAMAFACTORY_DIR}/data' \
        --generated_responses_dir '${IT_DIR}/responses' \
        --oracle_evaluations_dir '${IT_DIR}/oracle' \
        --merged_lora_root '${OUTPUT_ROOT}/${DATASET}'
"

# --- Step 4: Oracle Annotation ---
if [ "$SKIP_ORACLE" = false ]; then
    run_step "Oracle Annotation" "
        cd '${AL_DIR}' &&
        bash '${SCRIPTS_DIR}/run_oracle.sh' $ITERATION \
            --dataset $DATASET --dataset-type $DATASET --output_root '$OUTPUT_ROOT'
    "
else
    log "Skipping oracle"
fi

# --- Step 5: ICL Question & Response Generation ---
if [ "$SKIP_ICL" = false ]; then
    clear_gpu

    ICL_MODEL="$BASE_MODEL"
    ICL_ADAPTER=""
    if [ "$ITERATION" -gt 0 ]; then
        ICL_CKPT=$(ls -td "${PREV_IT_DIR}/model"/checkpoint-* 2>/dev/null | head -n 1 || true)
        if [ -n "$ICL_CKPT" ]; then
            ICL_ADAPTER="${ICL_CKPT//:/}"
        elif [ -d "${PREV_IT_DIR}/model/merged" ]; then
            ICL_MODEL="${PREV_IT_DIR}/model/merged"
        fi
    fi

    run_step "ICL Question & Response Generation" "
        cd '${AL_DIR}' &&
        bash '${SCRIPTS_DIR}/run_generate_questions_icl.sh' \
            $ITERATION $ORACLE_BUDGET $NUM_PROMPTS '$ICL_MODEL' '$ICL_ADAPTER' $DATASET '$OUTPUT_ROOT'
    "
    clear_gpu
else
    log "Skipping ICL generation"
fi

# --- Step 6: ICL Self-Consistency ---
temperatures=(0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70)
ICL_RESPONSE_FILES=""
for i in {1..8}; do
    ICL_RESPONSE_FILES="$ICL_RESPONSE_FILES ${IT_DIR}/icl/run${i}_t${temperatures[$((i-1))]}.jsonl"
done

run_step "Self-Consistency Analysis (ICL)" "
    cd '${AL_DIR}' &&
    python '${SCRIPTS_DIR}/self_consistency_icl.py' \
        --response_files $ICL_RESPONSE_FILES \
        --output_root '$OUTPUT_ROOT' \
        --dataset-prefix $DATASET \
        --iteration $ITERATION \
        --consistency_threshold $CONSISTENCY_THRESHOLD \
        --questions_file '${IT_DIR}/icl/questions.jsonl'
"

# --- Step 7: Create Final Preference Dataset ---
run_step "Create Preference Dataset" "
    python '${SCRIPTS_DIR}/create_preference_dataset.py' \
        --oracle_eval '${IT_DIR}/oracle/eval_oracle.json' \
        --high_conf_eval '${IT_DIR}/oracle/eval_high_conf.json' \
        --self_label '${IT_DIR}/consistency/self_label.json' \
        --icl_high_conf '${IT_DIR}/icl/high_consistent.json' \
        --output_root '$OUTPUT_ROOT' \
        --dataset $DATASET \
        --iteration $ITERATION
"

# Symlink preference dataset into LLaMA-Factory data dir
PREF_LF_NAME="${DATASET}_AL_it${ITERATION}_preference_pairs"
ln -sf "${IT_DIR}/training/preference_pairs.json" "${LLAMAFACTORY_DIR}/data/${PREF_LF_NAME}.json"

# --- Step 8: DPO Training ---
if [ "$SKIP_DPO" = false ]; then
    clear_gpu
    run_step "DPO Training" "
        bash '${SCRIPTS_DIR}/dpo_train.sh' \
            --dataset $DATASET --iteration $ITERATION --output_root '$OUTPUT_ROOT'
    "
else
    log "Skipping DPO training"
fi

# --- Step 9: Merge LoRA ---
run_step "Merge LoRA" "
    bash '${SCRIPTS_DIR}/merge_lora.sh' \
        --base_model '$BASE_MODEL' \
        --checkpoint_dir '${IT_DIR}/model' \
        --template '$TEMPLATE' \
        --export_dir '${IT_DIR}/model/merged'
"

log "Pipeline complete: dataset=$DATASET iteration=$ITERATION"
log "Outputs: $IT_DIR"
log "Next: bash run_iteration.sh $((ITERATION + 1)) --dataset $DATASET"
