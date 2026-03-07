import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gorillawatch.model.basemodel import TimmWrapper

def generate_embeddings(model, data_loader, use_amp=True):
    embeddings_tensor = torch.tensor([], device=model.device)
    labels_tensor = torch.tensor([], device=model.device)
    videos_tensor = torch.tensor([], device=model.device)
    dates_tensor = torch.tensor([], device=model.device)
    cameras_tensor = torch.tensor([], device=model.device)

    model.eval()
    print("Generating embeddings...")
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_amp):
        for images, labels, videos, dates, cameras in tqdm(data_loader):
            images = images.to(model.device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.clone().detach().to(model.device)
            videos = videos.clone().detach().to(model.device)
            dates = dates.clone().detach().to(model.device)
            cameras = cameras.clone().detach().to(model.device)

            embeddings = model(images)
            embeddings_tensor = torch.cat([embeddings_tensor, embeddings])
            labels_tensor = torch.cat([labels_tensor, labels])
            videos_tensor = torch.cat([videos_tensor, videos])
            dates_tensor = torch.cat([dates_tensor, dates])
            cameras_tensor = torch.cat([cameras_tensor, cameras])

    return embeddings_tensor, labels_tensor, videos_tensor, dates_tensor, cameras_tensor

def calculate_distance(embeddings, test_embedding, metric):
    if metric == "euclidean":
        distance = torch.norm(embeddings - test_embedding, dim=1)
    elif metric == "cosine":
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_test_embedding = F.normalize(test_embedding, p=2, dim=0)
        cosine_similarity = torch.matmul(normalized_embeddings, normalized_test_embedding.transpose(0, 1)).squeeze()
        distance = 1 - cosine_similarity
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return distance

def knn_cross_encounter(query_embeddings, query_images, query_labels, query_video_ids, query_date_ids, query_camera_ids, k, gallery_loader, model, distance_metric="euclidean", split="unknown", cross_encounter=True):

    # set up queries for KNN (already on model.device from generate_embeddings)
    query_labels_tensor = torch.as_tensor(query_labels, device=query_embeddings.device)
    query_video_ids_tensor = torch.as_tensor(query_video_ids, device=query_embeddings.device)
    query_date_ids_tensor = torch.as_tensor(query_date_ids, device=query_embeddings.device)
    query_camera_ids_tensor = torch.as_tensor(query_camera_ids, device=query_embeddings.device)

    # set up gallery (database) for KNN retrieval
    gallery_embeddings, gallery_labels, gallery_video_ids, gallery_date_ids, gallery_camera_ids = generate_embeddings(model, gallery_loader, use_amp=True)
    # gallery_embeddings already on model.device
    labels_tensor = torch.as_tensor(gallery_labels, device=gallery_embeddings.device)
    video_ids_tensor = torch.as_tensor(gallery_video_ids, device=gallery_embeddings.device)
    date_ids_tensor = torch.as_tensor(gallery_date_ids, device=gallery_embeddings.device)
    camera_ids_tensor = torch.as_tensor(gallery_camera_ids, device=gallery_embeddings.device)

    print("Number of queries: ", len(query_images))
    print("Number of gallery embeddings: ", len(gallery_embeddings))

    num_predictions = len(query_images)
    predictions = torch.zeros(num_predictions, dtype=torch.long, device=query_embeddings.device)
    actuals = torch.zeros(num_predictions, dtype=torch.long, device=query_embeddings.device)

    num_neighbors = k

    # Track masking statistics
    masked_counts = []

    # iterate through each query embedding against the gallery embeddings
    for i, idx in enumerate(query_images):

        test_embedding = query_embeddings[idx].unsqueeze(0)
        distances = calculate_distance(gallery_embeddings, test_embedding, distance_metric)

        # Mask out self and same video
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[idx] = False  # Exclude self
        mask &= (video_ids_tensor != query_video_ids_tensor[idx])  # Different videos

        # Cross-encounter masking: exclude same camera on same date (default)
        if cross_encounter:
            same_camera = (camera_ids_tensor == query_camera_ids_tensor[idx])
            same_date = (date_ids_tensor == query_date_ids_tensor[idx])
            mask &= ~(same_camera & same_date)  # Exclude same encounter (camera AND date)

        # Track masking statistics
        num_masked = (~mask).sum().item()
        num_available = mask.sum().item()
        masked_counts.append(num_masked)

        # Get top k neighbor indices
        top_k_values, top_k_indices = torch.topk(distances[mask], k=min(num_neighbors, mask.sum().item()), largest=False)
        neighbor_indices = torch.nonzero(mask)[top_k_indices].squeeze(1)

        if len(neighbor_indices) > 0:
            neighbor_labels = labels_tensor[neighbor_indices]
            predicted_label = torch.mode(neighbor_labels)[0]
            predictions[i] = predicted_label
            actuals[i] = query_labels_tensor[idx]

    # Log masking statistics
    import numpy as np
    import wandb

    masked_counts_np = np.array(masked_counts)
    masking_strategy = "Cross-encounter (camera+date)" if cross_encounter else "Cross-video"
    print(f"\n{masking_strategy} masking statistics:")
    print(f"  Samples masked per query - Min: {masked_counts_np.min()}, Max: {masked_counts_np.max()}, "
          f"Mean: {masked_counts_np.mean():.1f}, Median: {np.median(masked_counts_np):.1f}")
    print(f"  Total gallery size: {len(gallery_embeddings)}, Avg available neighbors: {len(gallery_embeddings) - masked_counts_np.mean():.1f}")

    # Log per-sample masking statistics to wandb
    if wandb.run is not None:
        # Create a table with per-query masking data
        masking_data = []
        for i, (num_masked, pred, actual) in enumerate(zip(masked_counts, predictions.cpu().numpy(), actuals.cpu().numpy())):
            masking_data.append([
                i,  # query index
                num_masked,  # samples masked
                len(gallery_embeddings) - num_masked,  # samples available
                int(pred),  # prediction
                int(actual),  # ground truth
                int(pred == actual)  # correct
            ])

        masking_table = wandb.Table(
            columns=["query_idx", "masked_count", "available_count", "prediction", "actual", "correct"],
            data=masking_data
        )
        masking_key = "cross_encounter_masking" if cross_encounter else "cross_video_masking"
        wandb.log({f"{split}/{masking_key}_per_query": masking_table})

        # Log summary statistics
        wandb.log({
            f"{split}/masking/min": int(masked_counts_np.min()),
            f"{split}/masking/max": int(masked_counts_np.max()),
            f"{split}/masking/mean": float(masked_counts_np.mean()),
            f"{split}/masking/median": float(np.median(masked_counts_np)),
            f"{split}/masking/gallery_size": len(gallery_embeddings),
            f"{split}/masking/avg_available": float(len(gallery_embeddings) - masked_counts_np.mean()),
        })

    # Calculate micro-average accuracy (across all samples)
    micro_accuracy = (predictions == actuals).float().mean().item()

    # Calculate macro-average accuracy (across all identities)
    unique_labels = torch.unique(actuals)
    per_class_accuracies = []
    for label in unique_labels:
        mask = actuals == label
        if mask.sum() > 0:
            class_accuracy = (predictions[mask] == actuals[mask]).float().mean().item()
            per_class_accuracies.append(class_accuracy)
    macro_accuracy = np.mean(per_class_accuracies) if per_class_accuracies else 0.0

    print(f"\nCorrect predictions: {micro_accuracy * len(predictions):.0f}/{len(predictions)}")
    print(f"Micro-average accuracy (all samples): {micro_accuracy:.4f}")
    print(f"Macro-average accuracy (per identity): {macro_accuracy:.4f}")
    print(f"Number of identities evaluated: {len(unique_labels)}")

    # Log both metrics to wandb
    wandb.log({
        f"{split}/accuracy/micro": micro_accuracy,
        f"{split}/accuracy/macro": macro_accuracy,
        f"{split}/accuracy/num_identities": len(unique_labels),
    })

    return micro_accuracy, macro_accuracy
    
# Evaluate a model using Cross-Video KNN classification
def evaluate_model(model_path, query_loader, query_dataset, config, gallery_loader, split="unknown", verbose=False, cross_encounter=True):
    # Check if this is zero-shot (no fine-tuned model) or fine-tuned evaluation
    is_zero_shot = not (model_path and model_path.strip() and os.path.exists(model_path))

    # For zero-shot: use identity layer (just backbone features)
    # For fine-tuned: use linear projection layer (as trained)
    if is_zero_shot:
        embedding_size = 256  # Not used with identity
        embedding_id = "identity"  # No additional layer for zero-shot
        print("Zero-shot evaluation: Using backbone features directly (no embedding layer)")
    else:
        embedding_size = 256
        embedding_id = "linear"
        print(f"Fine-tuned evaluation: Using trained linear embedding layer ({embedding_size}D)")

    dropout_p = 0.0
    pool_mode = "none"
    embedding_model = TimmWrapper(
        backbone_name=config.backbone_name,
        embedding_size=embedding_size,
        embedding_id=embedding_id,
        dropout_p=dropout_p,
        pool_mode=pool_mode,
        img_size=config.image_size,
    ).to(config.device)

    # Load the model weights only if a valid path is provided
    if not is_zero_shot:
        state_dict = torch.load(model_path, map_location=embedding_model.device, weights_only=False)
        # remove the prefix "module." from the keys of the state_dict
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        embedding_model.load_state_dict(state_dict)
    embedding_model = embedding_model.to(memory_format=torch.channels_last)
    embedding_model.eval()

    with torch.amp.autocast(device_type='cuda'):
        query_embeddings, query_labels, query_video_ids, query_date_ids, query_camera_ids = generate_embeddings(embedding_model, query_loader, use_amp=True)
        micro_accuracy, macro_accuracy = knn_cross_encounter(query_embeddings, query_dataset.images_for_cv_knn, query_labels, query_video_ids, query_date_ids, query_camera_ids, config.k, gallery_loader, embedding_model, distance_metric="euclidean", split=split, cross_encounter=cross_encounter)

    if verbose:
        print(f"Cross-Video KNN{config.k} Micro Accuracy: {micro_accuracy:.4f}")
        print(f"Cross-Video KNN{config.k} Macro Accuracy: {macro_accuracy:.4f}")

    return micro_accuracy, macro_accuracy