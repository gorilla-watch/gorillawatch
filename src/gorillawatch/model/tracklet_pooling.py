import torch
from collections import defaultdict
import numpy as np


def pool_embeddings_by_tracklet(embeddings, labels, video_ids, dates, cameras, pooling_method="average"):
    """
    Pool embeddings by tracklet (same label + same video).
    
    Args:
        embeddings: Tensor of shape (N, embedding_dim)
        labels: Tensor of shape (N,) with class labels
        video_ids: Tensor of shape (N,) with video IDs
        dates: Tensor of shape (N,) with date IDs
        cameras: Tensor of shape (N,) with camera IDs
        pooling_method: "average", "max", or "median"
    
    Returns:
        pooled_embeddings: Tensor of shape (num_tracklets, embedding_dim)
        tracklet_labels: List of class labels for each tracklet
        tracklet_video_ids: List of video IDs for each tracklet
        tracklet_dates: List of date IDs for each tracklet (taken from first image)
        tracklet_cameras: List of camera IDs for each tracklet (taken from first image)
        tracklet_image_indices: List of lists, indices of images in each tracklet
    """
    device = embeddings.device
    
    # Ensure tensors are on CPU for grouping
    labels_cpu = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    video_ids_cpu = video_ids.cpu().numpy() if isinstance(video_ids, torch.Tensor) else video_ids
    dates_cpu = dates.cpu().numpy() if isinstance(dates, torch.Tensor) else dates
    cameras_cpu = cameras.cpu().numpy() if isinstance(cameras, torch.Tensor) else cameras
    
    # Group by tracklet (label, video)
    tracklets = defaultdict(list)  # (label, video) -> list of indices
    for idx in range(len(embeddings)):
        tracklet_key = (int(labels_cpu[idx]), int(video_ids_cpu[idx]))
        tracklets[tracklet_key].append(idx)
    
    # Pool embeddings for each tracklet
    pooled_embeddings_list = []
    tracklet_labels = []
    tracklet_video_ids = []
    tracklet_dates = []
    tracklet_cameras = []
    tracklet_image_indices = []
    
    for (label, video_id), indices in sorted(tracklets.items()):
        indices_tensor = torch.tensor(indices, device=device)
        tracklet_embeddings = embeddings[indices_tensor]
        
        if pooling_method == "average":
            pooled_emb = tracklet_embeddings.mean(dim=0)
        elif pooling_method == "max":
            pooled_emb = tracklet_embeddings.max(dim=0)[0]
        elif pooling_method == "median":
            pooled_emb = torch.median(tracklet_embeddings, dim=0)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        pooled_embeddings_list.append(pooled_emb)
        tracklet_labels.append(label)
        tracklet_video_ids.append(video_id)
        # Take date and camera from first image in tracklet
        tracklet_dates.append(int(dates_cpu[indices[0]]))
        tracklet_cameras.append(int(cameras_cpu[indices[0]]))
        tracklet_image_indices.append(indices)
    
    pooled_embeddings = torch.stack(pooled_embeddings_list)
    
    return (
        pooled_embeddings,
        tracklet_labels,
        tracklet_video_ids,
        tracklet_dates,
        tracklet_cameras,
        tracklet_image_indices,
    )


def get_pooling_method_fn(method_name="average"):
    """Get the pooling function by name."""
    methods = {
        "average": lambda x: x.mean(dim=0),
        "max": lambda x: x.max(dim=0)[0],
        "median": lambda x: torch.median(x, dim=0)[0],
    }
    if method_name not in methods:
        raise ValueError(f"Unknown pooling method: {method_name}. Available: {list(methods.keys())}")
    return methods[method_name]
