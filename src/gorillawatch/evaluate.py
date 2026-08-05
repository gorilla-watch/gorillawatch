# Model Training and Evaluation
import os
import argparse
import time
import wandb
from torch.utils.data import ConcatDataset, DataLoader

from data_hf.data_loading import prepare_data
from model.model_evaluation import evaluate_model
from utils.determinism_helper import set_deterministic_seeds


def parse_args():
    "Parse command-line arguments for evaluation"
    argparser = argparse.ArgumentParser(description='Process evaluation parameters')
    argparser.add_argument('--wandb_entity', type=str, default="gorillawatch", help='wandb entity')
    argparser.add_argument('--wandb_project', type=str, default="New Pipeline", help='wandb project')
    argparser.add_argument('--wandb_run', type=str, default="best possible run", help='wandb run name')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')
    argparser.add_argument('--device', type=str, default="cuda", help='device to use (cuda or cpu)')
    argparser.add_argument('--dataset', type=str, default="gorilla-watch/Gorilla-SPAC-Wild", help='data path (HuggingFace dataset id)')
    argparser.add_argument('--dataset_config', type=str, default="face")
    argparser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    argparser.add_argument('--train_val_ratio', type=float, default=0.2, help='train/val split ratio')
    argparser.add_argument('--batch_size', type=int, default=8, help='batch size')
    argparser.add_argument('--image_size', type=int, default=224, help='image size')
    argparser.add_argument('--k', type=int, default=5, help='number of neighbors')
    argparser.add_argument('--backbone_name', type=str, default="vit_small_patch14_dinov2.lvd142m", help='backbone model name from timm')
    argparser.add_argument('--evaluate_model_path', type=str, default="", help='path to fine-tuned checkpoint (optional, uses pre-trained if not provided)')
    argparser.add_argument('--cross_video_only', action='store_true', help='use cross-video masking only (default is cross-encounter: same camera+date)')
    argparser.add_argument('--pool_tracklets', action='store_true', help='pool embeddings by tracklet (class+video) for evaluation')
    argparser.add_argument('--pooling_method', type=str, default='average', choices=['average', 'max', 'median'], help='pooling method for tracklets')
    return argparser.parse_args()

def setup_paths(config):
    """Make sure needed paths exist (only if model path is provided)"""
    if config.evaluate_model_path and config.evaluate_model_path.strip():
        os.makedirs(os.path.dirname(config.evaluate_model_path), exist_ok=True)

def main():
    total_start = time.time()
    print("Starting evaluation script")
    config = parse_args()
    print(
        "Config loaded: "
        f"run={config.wandb_run}, backbone={config.backbone_name}, "
        f"model_path={'set' if config.evaluate_model_path else 'none'}, "
        f"batch_size={config.batch_size}, workers={config.num_workers}, image_size={config.image_size}, k={config.k}, "
        f"pool_tracklets={config.pool_tracklets}, pooling_method={config.pooling_method if config.pool_tracklets else 'N/A'}"
    )

    print("Setting deterministic seeds")
    set_deterministic_seeds(seed=config.seed)
    print("Ensuring output/checkpoint paths exist")
    setup_paths(config)

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run, 
        settings=wandb.Settings(git_remote_url=None),
        config=vars(config)
    )

    # Log model info
    if config.evaluate_model_path and config.evaluate_model_path.strip() and os.path.exists(config.evaluate_model_path):
        print(f"Using checkpoint weights from: {config.evaluate_model_path}")
        artifact = wandb.Artifact('used_model', type='model')
        artifact.add_file(config.evaluate_model_path)  
        run.log_artifact(artifact)
        wandb.log({"model/name": f"{config.evaluate_model_path}"})
    else:
        print(f"Using pre-trained backbone weights from timm: {config.backbone_name}")
        wandb.log({"model/name": f"Pre-trained: {config.backbone_name}"})

    print("Preparing data splits and dataloaders (this can take time on first run)")
    data_start = time.time()
    data_splits = prepare_data(
        data_path=config.dataset,
        dataset_config=config.dataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        image_size=config.image_size,
        k=config.k,
        parallelize=False,
    )
    print(f"Data preparation completed in {time.time() - data_start:.1f}s")

    # Extract datasets for evaluation
    splits_to_evaluate = {}
    if "val" in data_splits:
        splits_to_evaluate["val"] = data_splits["val"]
    if "test" in data_splits:
        splits_to_evaluate["test"] = data_splits["test"]

    for split, (dataset, loader) in splits_to_evaluate.items():
        print(f"Preparing gallery (retrieval database) for split='{split}'")

        concat_datasets = [dataset]
        if "train" in data_splits:
            concat_datasets.append(data_splits["train"][0])
        else:
            print("Warning: 'train' split is not available in dataset")

        if "single_encounter" in data_splits:
            concat_datasets.append(data_splits["single_encounter"][0])
        else:
            print("Warning: 'single_encounter' split is not available in dataset")


        gallery_dataset = ConcatDataset(concat_datasets)
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        split_samples = len(dataset)
        split_batches = len(loader)
        gallery_batches = len(gallery_loader)
        
        print(
            f"Starting split='{split}' evaluation "
            f"(query_samples={split_samples}, query_batches={split_batches}, gallery_batches={gallery_batches})"
        )
        
        split_start = time.time()
        cross_encounter = not config.cross_video_only  # Default is cross-encounter, flag disables it
        micro_accuracy, macro_accuracy, tracklet_micro_accuracy, tracklet_macro_accuracy = evaluate_model(
            config.evaluate_model_path, loader, dataset, config, gallery_loader, split=split, verbose=True, 
            cross_encounter=cross_encounter, pool_tracklets=config.pool_tracklets, 
            pooling_method=config.pooling_method
        )
        split_duration = time.time() - split_start
        print(f"Finished split='{split}' in {split_duration:.1f}s")
        print(f"  Image-based Micro accuracy: {micro_accuracy:.4f}")
        print(f"  Image-based Macro accuracy: {macro_accuracy:.4f}")
        if config.pool_tracklets:
            print(f"  Tracklet-based Micro accuracy: {tracklet_micro_accuracy:.4f}")
            print(f"  Tracklet-based Macro accuracy: {tracklet_macro_accuracy:.4f}")
        wandb.log({
            f"{split}/accuracy/micro": micro_accuracy,
            f"{split}/accuracy/macro": macro_accuracy,
            f"{split}/duration": split_duration
        })
        if config.pool_tracklets:
            wandb.log({
                f"{split}/tracklet/accuracy/micro": tracklet_micro_accuracy,
                f"{split}/tracklet/accuracy/macro": tracklet_macro_accuracy,
            })

    total_duration = time.time() - total_start
    print(f"Evaluation script completed in {total_duration:.1f}s")
    wandb.finish()

if __name__ == "__main__":
    main()
