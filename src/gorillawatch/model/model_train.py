import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm  
from logging import getLogger
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import time

from gorillawatch.model.basemodel import TimmWrapper
from gorillawatch.losses.triplet_loss import TripletLossOnline
from gorillawatch.losses.regularization import L2SPRegularization_Wrapper
from gorillawatch.model.model_evaluation import evaluate_model


logger = getLogger(__name__)


def train_epoch_amp(model, loader, scaler, loss_fn, optimizer, device, config, cross_video_masking=False):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()  # Zero gradients outside the accumulation loop
    
    # Handle DDP loader lengths only if DDP is enabled
    if config.use_ddp:
        # Gather lengths of loaders from all ranks and find the minimum length
        local_loader_length = len(loader)
        loader_lengths = [torch.tensor(local_loader_length, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(loader_lengths, loader_lengths[dist.get_rank()])
        min_loader_length = min(loader_lengths).item()
        
        if local_loader_length != min_loader_length:
            error = ValueError(f"Rank {device} has loader length {local_loader_length} while the smallest loader length is {min_loader_length} epochs")
            print(f"An error occurred: {error}")
            dist.destroy_process_group()
    else:
        min_loader_length = len(loader)

    for batch_idx, (images, labels, videos, dates, cameras) in enumerate(tqdm(loader)):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type="cuda"):
            embeddings = model(images)
            if cross_video_masking:
                loss, pos_dist, neg_dist = loss_fn(embeddings, labels, ids=videos)
            else:
                loss, pos_dist, neg_dist = loss_fn(embeddings, labels)
        
        # Scale the loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        scaler.scale(loss).backward()

        # Optionally, add the unscaled loss for logging purposes
        epoch_loss += loss.item() * config.gradient_accumulation_steps

        # Only update the weights after GRADIENT_ACCUMULATION_STEPS mini-batches
        if ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Compute the average loss - handle differently depending on DDP or not
    if config.use_ddp:
        total_loss = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss /= dist.get_world_size()
        return total_loss.item() / min_loader_length
    else:
        return epoch_loss / min_loader_length


def validate_epoch(model, loader, loss_fn, device, cross_video_masking=False):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, labels, videos, dates, cameras in loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            if cross_video_masking:
                loss, pos_dist, neg_dist = loss_fn(embeddings, labels, ids=videos)
            else:
                loss, pos_dist, neg_dist = loss_fn(embeddings, labels)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def train_and_val_model(train_loader, train_val_loader, val_loader, val_dataset, rank, config, gallery_loader, verbose=True, pre_evaluation=False, wandb_run=None):
    embedding_size = 256  # Desired embedding size
    dropout_p = 0.0  # Dropout probability for embedding layer
    embedding_id = "linear"  # Type of embedding layer ("linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout")
    pool_mode = "none"  # Global pooling method ("gem", "gap", "gem_c", "none")
    scaler = torch.amp.GradScaler()
    triplet_loss_fn = TripletLossOnline(margin=config.margin, mode="hard", dist_calc="euclidean", cross_video_masking=False)
     
    # Initialize Timm Wrapper Model
    embedding_model = TimmWrapper(
        backbone_name=config.backbone_name,
        embedding_size=embedding_size,
        embedding_id=embedding_id,
        dropout_p=dropout_p,
        pool_mode=pool_mode,
        load_pretrained=config.load_pretrained,
        pretrained_weights_path=config.pretrained_weights_path,
        img_size=config.image_size,
    ).to(rank)
    
    # Only wrap with DDP if it's enabled
    if config.use_ddp:
        embedding_model = DDP(embedding_model, device_ids=[rank])
    
    l2sp_loss_fn = L2SPRegularization_Wrapper(
        loss=triplet_loss_fn,
        model=embedding_model.module.model if config.use_ddp else embedding_model.model,
        path_to_pretrained_weights=config.pretrained_weights_path,
        alpha=config.l2sp_reg,
        beta=config.l2_reg,
    )
    # Optimizer
    optimizer = AdamW(
        embedding_model.parameters(),
        lr=config.lr * (torch.cuda.device_count() if config.use_ddp else 1),
        betas=(0.9, 0.999),   # beta1, beta2
        eps=1e-7              # epsilon
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,     # total number of epochs to run
        eta_min=1e-7
    )
    # Early Stopping
    best_val_loss = float('inf')
    val_loss = 0
    patience_counter = 0

    if rank == 0:
        torch.save(
            embedding_model.module.state_dict() if config.use_ddp else embedding_model.state_dict(),
            config.best_model_path
        )
        run_start_time = time.time()
        epoch_start_time = time.time()
        if verbose and pre_evaluation:
            print(f"Train_val loss before training: {validate_epoch(embedding_model, train_val_loader, l2sp_loss_fn, rank):.4f}")
            print(f"Validation loss before training: {validate_epoch(embedding_model, val_loader, l2sp_loss_fn, rank):.4f}")
    
    if config.use_ddp:
        dist.barrier()
        
    epoch = 0
    for epoch in range(config.epochs):
        if config.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        if verbose:
            print(f"Epoch {epoch + 1}/{config.epochs}")
        
        train_loss = train_epoch_amp(embedding_model, train_loader, scaler, l2sp_loss_fn, optimizer, rank, config, False)
        if rank == 0:
            val_loss = validate_epoch(embedding_model, train_val_loader, l2sp_loss_fn, rank, False)
        
        scheduler.step()
        
        if config.use_ddp:
            dist.barrier()
        
        if verbose and rank == 0:
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, epoch_time: {time.time()-epoch_start_time:.2f}s")
            epoch_start_time = time.time()
            
        if wandb_run and rank == 0:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "val/best_loss": best_val_loss,
            })  
        
        if val_loss < best_val_loss:
            if rank == 0:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    embedding_model.module.state_dict() if config.use_ddp else embedding_model.state_dict(),
                    config.best_model_path
                )
        else:
            patience_counter += 1
        
        # Broadcast patience counter only if using DDP
        if config.use_ddp:
            patience_counter_tensor = torch.tensor(patience_counter, device=rank)
            dist.broadcast(patience_counter_tensor, src=0)
            patience_counter = patience_counter_tensor.item()

        if patience_counter >= config.early_stopping_patience:
            if rank == 0:
                print("Early stopping!")
            break

        # Periodic evaluation during training
        if config.eval_frequency > 0 and (epoch + 1) % config.eval_frequency == 0 and rank == 0:
            if verbose:
                print(f"Running evaluation at epoch {epoch + 1}...")
            eval_micro, eval_macro, _, _ = evaluate_model(
                config.best_model_path,
                val_loader,
                val_dataset,
                config,
                gallery_loader=gallery_loader,
                split="val",
                verbose=False
            )
            if verbose:
                print(f"Epoch {epoch + 1} Evaluation - Micro: {eval_micro:.4f}, Macro: {eval_macro:.4f}")
            if wandb_run:
                wandb.log({
                    "val/accuracy/micro": eval_micro,
                    "val/accuracy/macro": eval_macro,
                })

    micro_accuracy, macro_accuracy, _, _ = evaluate_model(config.best_model_path, val_loader, val_dataset, config, gallery_loader=gallery_loader, split="val", verbose=verbose)

    if rank == 0:
        if verbose:
            if config.epochs > 0:
                print("Time spent:", time.time()-run_start_time, "average time per epoch:", (time.time()-run_start_time)/config.epochs)

        if wandb_run:
            wandb.log({
                "val/final/accuracy/micro": micro_accuracy,
                "val/final/accuracy/macro": macro_accuracy,
                "train/best_model_path": config.best_model_path,
                "train/total_epochs": epoch + 1
            })
    
    if config.use_ddp:
        dist.barrier()

    return micro_accuracy, macro_accuracy
