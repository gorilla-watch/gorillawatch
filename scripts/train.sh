#!/bin/bash
# Fine-tune ViT-Giant DINOv2 model on Gorilla-SPAC-Wild dataset

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
BACKBONE="vit_giant_patch14_dinov2.lvd142m"
DATASET="gorilla-watch/Gorilla-SPAC-Wild"
DATASET_CONFIG="face_with_body"

IMAGE_SIZE=518
BATCH_SIZE=8
NUM_WORKERS=4
K=5
SEED=42
EPOCHS=100
LR=0.00000019
GRADIENT_ACCUMULATION_STEPS=6
EVAL_FREQUENCY=10  # Evaluate every N epochs (0 = only at end)

# Paths
BEST_MODEL_PATH="saved_checkpoints/${BACKBONE}_fine_tuned.pth"
PRETRAINED_WEIGHTS_PATH="pretrained_weights/${BACKBONE}.pth"

# WandB configuration
WANDB_ENTITY=""
WANDB_PROJECT=""
WANDB_RUN=""

echo "========================================"
echo "Fine-tuning Gorilla Re-ID Model"
echo "========================================"
echo "Backbone: $BACKBONE"
echo "Dataset: $DATASET"
echo "Config: $DATASET_CONFIG"
echo "Image size: $IMAGE_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "Model will be saved to: $BEST_MODEL_PATH"
echo ""

# Run training
python src/gorillawatch/train_and_eval.py \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run "$WANDB_RUN" \
    --seed $SEED \
    --dataset "$DATASET" \
    --dataset_config "$DATASET_CONFIG" \
    --num_workers $NUM_WORKERS \
    --best_model_path "$BEST_MODEL_PATH" \
    --backbone_name "$BACKBONE" \
    --pretrained_weights_path "$PRETRAINED_WEIGHTS_PATH" \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --k $K \
    --epochs $EPOCHS \
    --early_stopping_patience 100 \
    --eval_frequency $EVAL_FREQUENCY \
    --use_ddp False

echo ""
echo "========================================"
echo "Training completed!"
echo "Model saved to: $BEST_MODEL_PATH"
echo "========================================"