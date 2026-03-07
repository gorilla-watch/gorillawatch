#!/bin/bash
# Evaluate fine-tuned ViT-Small DINOv2 model

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
BACKBONE="vit_small_patch14_dinov2.lvd142m"
CHECKPOINT_PATH="saved_checkpoints/vit_small_patch14_dinov2.lvd142m_fine_tuned.pth"
DATASET="gorilla-watch/Gorilla-SPAC-Wild"
DATASET_CONFIG="face"
IMAGE_SIZE=224
BATCH_SIZE=8
NUM_WORKERS=4
K=5
SEED=42
TRAIN_VAL_RATIO=0.2

# WandB configuration
WANDB_ENTITY="gorillawatch"
WANDB_PROJECT="GorillaWatch-Evaluation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WANDB_RUN="vit_small_fine_tuned_${TIMESTAMP}"

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
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Run evaluation
python src/gorillawatch/evaluate.py \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run "$WANDB_RUN" \
    --seed $SEED \
    --dataset "$DATASET" \
    --dataset_config "$DATASET_CONFIG" \
    --num_workers $NUM_WORKERS \
    --train_val_ratio $TRAIN_VAL_RATIO \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --k $K \
    --backbone_name "$BACKBONE" \
    --evaluate_model_path "$CHECKPOINT_PATH" 2>&1 | tee "$RESULTS_DIR/eval_${TIMESTAMP}.log"

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results logged to WandB and saved to: $RESULTS_DIR/eval_${TIMESTAMP}.log"
echo "========================================"
