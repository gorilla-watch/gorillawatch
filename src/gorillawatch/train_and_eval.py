# Model Training and Evaluation
import os
import argparse
import timm
import torch
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from data_hf.data_loading import prepare_data as prepare_data_hf
from data.data_loading import prepare_data as prepare_data_local
from model.model_train import train_and_val_model
from utils.determinism_helper import set_deterministic_seeds
from torch.utils.data import ConcatDataset, DataLoader

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--wandb_entity', type=str, default="gorillawatch", help='wandb entity')
    argparser.add_argument('--wandb_project', type=str, default="New Pipeline", help='wandb project')
    argparser.add_argument('--wandb_run', type=str, default="best possible run", help='wandb run name')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')
    argparser.add_argument('--device', type=str, default="cuda", help='device to use (cuda or cpu)')
    argparser.add_argument('--dataset', type=str, default="gorilla-watch/Gorilla-SPAC-Wild", help='HuggingFace dataset id or local path')
    argparser.add_argument('--dataset_config', type=str, default="face", help='dataset configuration')
    argparser.add_argument('--split_path', type=str, default=None, help='path to JSON file with train/val/test splits (for local datasets)')
    argparser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    argparser.add_argument('--best_model_path', type=str, default="saved_checkpoints/20260304_120000.pth", help='best model path')
    argparser.add_argument('--backbone_name', type=str, default="vit_giant_patch14_dinov2.lvd142m", help='backbone name')
    argparser.add_argument('--load_pretrained', type=t_or_f, default=False, help='load pretrained weights')
    argparser.add_argument('--pretrained_weights_path', type=str, default="pretrained_weights/vit_giant_patch14_dinov2.lvd142m224.pth", help='pretrained weights path')
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=6, help='gradient accumulation steps')
    argparser.add_argument('--l2_reg', type=float, default=0.0059, help='L2 regularization')
    argparser.add_argument('--l2sp_reg', type=float, default=0.000013, help='L2SP regularization')
    argparser.add_argument('--image_size', type=int, default=518, help='image size')
    argparser.add_argument('--batch_size', type=int, default=8, help='batch size')
    argparser.add_argument('--lr', type=float, default=0.0000019, help='learning rate')
    argparser.add_argument('--margin', type=float, default=0.647, help='triplet margin')
    argparser.add_argument('--k', type=int, default=5, help='number of neighbors')
    argparser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--early_stopping_patience', type=int, default=100, help='early stopping patience')
    argparser.add_argument('--eval_frequency', type=int, default=0, help='evaluate every N epochs (0 = only at end)')
    argparser.add_argument('--evaluate_model_path', type=str, default="/workspaces/vast-gorilla/gorillawatch/models/ViT-giant-229epochs.pt", help='evaluate model path')
    argparser.add_argument('--use_ddp', type=t_or_f, default=False, help='use distributed data parallel')
    return argparser.parse_args()

def setup_paths(config):
    os.makedirs(os.path.dirname(config.best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.pretrained_weights_path), exist_ok=True)
    
    if os.path.exists(config.best_model_path):
        print(f"Warning: Best model path {config.best_model_path} already exists. Finding unique name...")
        i = 0
        while True:
            new_path = f"{config.best_model_path}_{i}"
            if not os.path.exists(new_path):
                config.best_model_path = new_path
                print(f"Unique best model path found: {config.best_model_path}")
                break
            i += 1

    if not os.path.exists(config.pretrained_weights_path):
        model = timm.create_model(config.backbone_name, pretrained=True, drop_rate=0.0, img_size=config.image_size)
        torch.save(model.state_dict(), config.pretrained_weights_path)

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()

def is_local_path(path):
    """Detect if the given path is a local filesystem path rather than a HuggingFace dataset ID.
    
    A HuggingFace dataset ID has format 'org/dataset' (contains / but doesn't start with /).
    A local path either starts with / (absolute) or doesn't contain / (relative).
    """
    return path.startswith("/") or "/" not in path

def main(rank, world_size, config):
    config.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    
    if config.use_ddp:
        ddp_setup(rank, world_size)
    
    if not config.use_ddp or rank == 0:
        config.effective_batch_size = config.batch_size * (world_size if config.use_ddp else 1) * config.gradient_accumulation_steps

        run = wandb.init(
            project= config.wandb_project,
            entity= config.wandb_entity,
            name= config.wandb_run, 
            settings=wandb.Settings(git_remote_url=None),
            config=vars(config)
        )
    set_deterministic_seeds(config.seed)

    # Use appropriate data loading based on input type
    if is_local_path(config.dataset):
        if not config.use_ddp or rank == 0:
            print(f"Loading from local path: {config.dataset}")
        # Local path: use data/ directory loader
        data_result = prepare_data_local(
            data_path=config.dataset,
            split_path=config.split_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            train_val_ratio=0.2,
            seed=config.seed,
            image_size=config.image_size,
            k=config.k,
            verbose=(not config.use_ddp or rank==0),
            parallelize=config.use_ddp,
            perform_train_val_split=False,
        )
        # Normalize local loader output to dict format
        if len(data_result) == 8:  # with train_val split
            train_set, train_loader, _, _, val_set, val_loader, test_set, test_loader = data_result
        else:  # without train_val split
            train_set, train_loader, val_set, val_loader, test_set, test_loader = data_result
        data_splits = {
            "train": (train_set, train_loader),
            "val": (val_set, val_loader),
            "test": (test_set, test_loader),
        }
    else:
        if not config.use_ddp or rank == 0:
            print(f"Loading from HuggingFace: {config.dataset}")
        # HuggingFace dataset ID: use data_hf/ directory loader
        data_splits = prepare_data_hf(
            data_path=config.dataset,
            dataset_config=config.dataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            image_size=config.image_size,
            k=config.k,
            verbose=(not config.use_ddp or rank==0),
            parallelize=config.use_ddp,
        )

    # Extract datasets and loaders from dictionary
    train_set, train_loader = data_splits["train"]
    val_set, val_loader = data_splits["val"]

    # Build gallery dataset
    gallery_datasets = [val_set, train_set]
    if "single_encounter" in data_splits:
        gallery_datasets.append(data_splits["single_encounter"][0])
        
    gallery_dataset = ConcatDataset(gallery_datasets)
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    micro_accuracy, macro_accuracy = train_and_val_model(
        train_loader,
        val_loader,
        val_loader,
        val_set,
        rank,
        config,
        gallery_loader,
        verbose=(not config.use_ddp or rank==0),
        pre_evaluation=False,
        wandb_run=(run if (not config.use_ddp or rank==0) else None),
    )

    if not config.use_ddp or rank == 0:
        wandb.log({
            "val/final/accuracy/micro": micro_accuracy,
            "val/final/accuracy/macro": macro_accuracy
        })
        wandb.finish()
    
    
    if config.use_ddp:
        ddp_cleanup()

if __name__ == "__main__":
    config = parse_args()
    setup_paths(config)

    if config.use_ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, config), nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1, config=config)