#!/bin/bash
# Evaluate the fine-tuned ViT-Giant DINOv2 model from huggingface

set -e
set -o pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
# BACKBONE, CHECKPOINT_PATH, and IMAGE_SIZE will be omitted if a HuggingFace model is provided
HF_MODEL="gorilla-watch/GorillaWatch-DINOv2-Giant"

DATASET="gorilla-watch/Gorilla-SPAC-Wild" # or our Berlin Zoo dataset "gorilla-watch/Gorilla-Zoo-Berlin"
DATASET_CONFIG="face_with_body"

BATCH_SIZE=8
NUM_WORKERS=4
K=5
SEED=42
POOL_TRACKLETS=true
POOLING_METHOD="average"

# WandB configuration
WANDB_ENTITY=""
WANDB_PROJECT=""
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WANDB_RUN=""

# Create results directory
RESULTS_DIR="eval_results"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Evaluating Fine-Tuned Model"
echo "========================================"
echo "HuggingFace model: $HF_MODEL"
echo "Dataset: $DATASET"
echo "Config: $DATASET_CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "K (neighbors): $K"
echo "Tracklet pooling: $POOL_TRACKLETS"
if [ "$POOL_TRACKLETS" = true ]; then
    echo "Pooling method: $POOLING_METHOD"
fi
echo ""

# Run evaluation
EVAL_ARGS=(
    --wandb_entity "$WANDB_ENTITY"
    --wandb_project "$WANDB_PROJECT"
    --wandb_run "$WANDB_RUN"
    --seed "$SEED"
    --dataset "$DATASET"
    --dataset_config "$DATASET_CONFIG"
    --num_workers "$NUM_WORKERS"
    --batch_size "$BATCH_SIZE"
    --image_size "$IMAGE_SIZE"
    --k "$K"
    --hf_model "$HF_MODEL"
)

if [ "$POOL_TRACKLETS" = true ]; then
    EVAL_ARGS+=(--pool_tracklets --pooling_method "$POOLING_METHOD")
fi

python src/gorillawatch/evaluate.py "${EVAL_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/eval_${TIMESTAMP}.log"

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results logged to WandB and saved to: $RESULTS_DIR/eval_${TIMESTAMP}.log"
echo "========================================"
