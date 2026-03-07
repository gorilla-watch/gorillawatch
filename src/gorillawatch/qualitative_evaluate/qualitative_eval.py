import os
from types import SimpleNamespace
from gorillawatch.data.data_loading import prepare_data
from gorillawatch.utils.determinism_helper import set_deterministic_seeds
from qual_eval_helpers import qual_eval_model

"""Make sure the IMAGE_SIZE is consistent with the model input size"""

# Set seeds for reproducibility


# default config
default_config = SimpleNamespace(
    wandb_run="best possible run",
    seed=42,
    device="cuda",
    data_path="/workspaces/vast-gorilla/gorillawatch/data2/spac23+24-body_face-squared-deduplicated",
    num_workers=2,
    best_model_path="saved_checkpoints/20260304_120000.pth",
    backbone_name="vit_giant_patch14_dinov2.lvd142m",
    pretrained_weights_path="pretrained_weights/vit_giant_patch14_dinov2.lvd142m224.pth",
    image_size=518,
    batch_size=8,
    gradient_accumulation_steps=6,
    lr=0.00000193433844038508,
    l2_reg=0.005917359394186353,
    l2sp_reg=0.00001348344403439296,
    margin=0.6474187205468019,
    k=5,
    epochs=100,
    early_stopping_patience=5,
    evaluate_model_path="/workspaces/vast-gorilla/gorillawatch/models/ViT-giant-229epochs.pt",
    use_ddp=False,
)

if __name__ == "__main__":
    set_deterministic_seeds(default_config.seed)
    train_set, train_loader, train_val_set, train_val_loader, val_set, val_loader, test_set, test_loader = prepare_data(
        data_path=default_config.data_path,
        split_path=default_config.split_path,
        batch_size=default_config.batch_size,
        num_workers=default_config.num_workers,
        train_val_ratio=default_config.train_val_ratio,
        seed=default_config.seed,
        image_size=default_config.image_size,
        k=default_config.k,
        wandb_run=None,
        parallelize=False,
    )
    
    # visulize the val set in Voxel51
    DATA_PATH_51 = os.path.join(default_config.data_path, "val")
    
    # calculate and save results.
    evaluator = qual_eval_model(EVALUATE_MODEL_PATH, val_loader, val_set, DATA_PATH_51, default_config, verbose=True, visualize=True)

