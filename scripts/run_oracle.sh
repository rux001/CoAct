#!/bin/bash
# Oracle evaluation via GPT (Azure OpenAI).
# Usage: bash run_oracle.sh <iteration> --dataset <name> --dataset-type <type> [--output_root <dir>]

set -euo pipefail

ITERATION=${1:-0}; shift || true

DATASET="gsm8k"
DATASET_TYPE="gsm8k"
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)       DATASET="$2"; shift 2 ;;
        --dataset=*)     DATASET="${1#*=}"; shift ;;
        --dataset-type)  DATASET_TYPE="$2"; shift 2 ;;
        --dataset-type=*) DATASET_TYPE="${1#*=}"; shift ;;
        --output_root)   OUTPUT_ROOT="$2"; shift 2 ;;
        --output_root=*) OUTPUT_ROOT="${1#*=}"; shift ;;
        *) echo "Warning: Unknown option '$1' ignored"; shift ;;
    esac
done

DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AL_DIR="$( cd "${SCRIPTS_DIR}/.." && pwd )"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AL_DIR}/output}"

AZURE_API_KEY="${AZURE_API_KEY:?Set AZURE_API_KEY}"
AZURE_ENDPOINT="${AZURE_ENDPOINT:?Set AZURE_ENDPOINT}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2025-01-01-preview}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-gpt-5-chat}"

HIGH_CONSISTENT_FILE="${OUTPUT_ROOT}/${DATASET}/it${ITERATION}/consistency/high_conf_candidates.json"
ORACLE_FILE="${OUTPUT_ROOT}/${DATASET}/it${ITERATION}/consistency/oracle_candidates.json"

echo "Oracle evaluation: iteration=$ITERATION dataset=$DATASET type=$DATASET_TYPE"
echo "  high_conf: $HIGH_CONSISTENT_FILE"
echo "  oracle:    $ORACLE_FILE"

python "${SCRIPTS_DIR}/oracle.py" \
    --dataset-type "${DATASET_TYPE}" \
    --output_root "${OUTPUT_ROOT}" \
    --dataset "${DATASET}" \
    --iteration "$ITERATION" \
    --azure_api_key "${AZURE_API_KEY}" \
    --azure_endpoint "${AZURE_ENDPOINT}" \
    --azure_api_version "${AZURE_API_VERSION}" \
    --deployment_name "${DEPLOYMENT_NAME}" \
    --high_consistent_file "$HIGH_CONSISTENT_FILE" \
    --oracle_file "$ORACLE_FILE"

echo "Oracle evaluation complete."
