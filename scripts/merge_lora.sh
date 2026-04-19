#!/bin/bash

# Merge a trained LoRA adapter back into the base model using llamafactory-cli.
#
# Usage:
#   bash merge_lora.sh \
#       --base_model <base_model_path_or_hf_id> \
#       --checkpoint_dir <lora_checkpoint_or_dir> \
#       --export_dir <output_dir> \
#       [--template <alpaca|llama3>]
#
# Required env vars: LLAMAFACTORY_DIR
#
# Note: Do NOT use quantized base models or set quantization_bit when merging LoRA adapters.

set -euo pipefail

BASE_MODEL=""
CHECKPOINT_DIR=""
EXPORT_DIR=""
TEMPLATE="llama3"
EXPORT_DEVICE="${EXPORT_DEVICE:-auto}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model)     BASE_MODEL="$2"; shift 2 ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --export_dir)     EXPORT_DIR="$2"; shift 2 ;;
        --template)       TEMPLATE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$BASE_MODEL" ] || [ -z "$CHECKPOINT_DIR" ] || [ -z "$EXPORT_DIR" ]; then
    echo "Usage: $0 --base_model <model> --checkpoint_dir <ckpt> --export_dir <dir> [--template <tmpl>]"
    exit 1
fi

LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR to your LLaMA-Factory checkout}"

echo "=============================================="
echo "LLaMA LoRA Checkpoint Merger"
echo "=============================================="
echo "Base Model:      $BASE_MODEL"
echo "Checkpoint Dir:  $CHECKPOINT_DIR"
echo "Export Dir:      $EXPORT_DIR"
echo "Template:        $TEMPLATE"
echo "=============================================="

cd "$LLAMAFACTORY_DIR"

# Resolve the adapter path: direct checkpoint, latest checkpoint in dir, or the dir itself.
ADAPTER_PATH=""
if [ -d "$CHECKPOINT_DIR" ]; then
    if [[ "$CHECKPOINT_DIR" == *"checkpoint-"* ]]; then
        ADAPTER_PATH="$CHECKPOINT_DIR"
    else
        LATEST_CHECKPOINT=$(ls -t "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | head -n 1 || true)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            ADAPTER_PATH="${LATEST_CHECKPOINT//:/}"
        else
            ADAPTER_PATH="$CHECKPOINT_DIR"
        fi
    fi
else
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Resolved adapter: $ADAPTER_PATH"
mkdir -p "$EXPORT_DIR"

llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL" \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --template "$TEMPLATE" \
    --trust_remote_code true \
    --export_dir "$EXPORT_DIR" \
    --export_device "$EXPORT_DEVICE"

echo "LoRA merge complete. Merged model saved to: $EXPORT_DIR"
