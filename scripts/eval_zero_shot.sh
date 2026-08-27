#!/bin/bash
# Zero-shot evaluation on a variety of models with face_and_body config

set -e
#pip install -e .

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
BATCH_SIZE=16
NUM_WORKERS=4
IMAGE_SIZE=224
K=5
SEED=42

# Evaluation parameters
TRAIN_VAL_RATIO=0.2

# Create results directory
RESULTS_DIR="eval_results"
mkdir -p "$RESULTS_DIR"

# WandB configuration
WANDB_ENTITY=""
WANDB_PROJECT=""

# Models to evaluate
declare -a MODELS=(
    "efficientnetv2_rw_m.agc_in1k"
    "vit_giant_patch14_dinov2.lvd142m"
    "vit_large_patch14_dinov2.lvd142m"
    "vit_base_patch14_dinov2.lvd142m"
    "vit_small_patch8_224.dino"
    "tf_efficientnetv2_b3.in21k"
    "vit_small_patch14_dinov2.lvd142m"
    "resnet50.fb_ssl_yfcc100m_ft_in1k"
    "resnet18.tv_in1k"
    "inception_v3.tf_adv_in1k"
    "vit_large_patch14_clip_224.openai"
    "swinv2_base_window12_192.ms_in22k"
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
    "swinv2_small_window16_256.ms_in1k"
)

# Function to get the appropriate image size for a model
get_image_size() {
    local model=$1
    # SwinV2 models require specific image sizes based on their window configuration
    if [[ "$model" == *"swinv2_base_window12_192"* ]]; then
        echo "192"
    elif [[ "$model" == *"swinv2"*"192to256"* ]] || [[ "$model" == *"swinv2"*"window16_256"* ]]; then
        echo "256"
    else
        echo "$IMAGE_SIZE"  # Default to 224 for other models
    fi
}

echo "========================================"
echo "Zero-Shot Evaluation on Small DINO Models"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Default image size: $IMAGE_SIZE"
echo "Number of workers: $NUM_WORKERS"
echo "K (neighbors): $K"
echo ""

# Iterate through models
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $model"
    echo "=========================================="

    # Get appropriate image size for this model
    MODEL_IMAGE_SIZE=$(get_image_size "$model")
    echo "Using image size: $MODEL_IMAGE_SIZE"

    # Create timestamped run name
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_NAME="${model//./_}_${TIMESTAMP}"

    # Run evaluation
    python src/gorillawatch/evaluate.py \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run "$RUN_NAME" \
        --seed $SEED \
        --dataset "gorilla-watch/Gorilla-SPAC-Wild" \
        --dataset_config "face_with_body" \
        --num_workers $NUM_WORKERS \
        --train_val_ratio $TRAIN_VAL_RATIO \
        --batch_size $BATCH_SIZE \
        --image_size $MODEL_IMAGE_SIZE \
        --k $K \
        --backbone_name "$model" 2>&1 | tee "$RESULTS_DIR/${model}_${TIMESTAMP}.log"
    
    echo "Completed: $model"
    echo ""
done

echo "========================================"
echo "All evaluations completed!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
