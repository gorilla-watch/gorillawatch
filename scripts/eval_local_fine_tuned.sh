#!/bin/bash
# Evaluate a fine-tuned ViT-Giant DINOv2 model from a local .pth checkpoint

set -e
set -o pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
BACKBONE="vit_large_patch14_dinov2.lvd142m" 
CHECKPOINT_PATH="saved_checkpoints/vit_large_patch14_dinov2.lvd142m_fine_tuned.pth"
IMAGE_SIZE=518

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
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Backbone: $BACKBONE"
echo "Dataset: $DATASET"
echo "Config: $DATASET_CONFIG"
echo "Image size: $IMAGE_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "K (neighbors): $K"
echo "Tracklet pooling: $POOL_TRACKLETS"
if [ "$POOL_TRACKLETS" = true ]; then
    echo "Pooling method: $POOLING_METHOD"
fi
echo ""

 Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

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
    --backbone_name "$BACKBONE"
    --evaluate_model_path "$CHECKPOINT_PATH"
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
