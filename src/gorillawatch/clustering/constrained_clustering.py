import os
import time
import torch
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset 
import numpy as np
from PIL import Image
from collections import Counter


from gorillawatch.model.basemodel import TimmWrapper
from gorillawatch.clustering.distance_matrix import get_distance_matrix

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self):
        image_paths = [
            os.path.join(self.root_dir, file) for file 
            in os.listdir(self.root_dir)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))]
        distinct_prefixes = set([os.path.basename(path)[:4] for path in image_paths])
        distinct_prefixes = sorted(distinct_prefixes, key=lambda x: x[::-1])
        print(f"Distinct 4-character prefixes: {distinct_prefixes}")
        prefix_counter = Counter([os.path.basename(path)[:4] for path in image_paths])
        count = len(prefix_counter)
        min_count = min(prefix_counter.values())
        print(f"Minimum count of images with the same 4 beginning characters: {min_count}, Expected Number of Clusters: {count}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path
    
    
def load_vit_model(checkpoint_path, device):
    embedding_size = 256  
    dropout_p = 0.0  
    embedding_id = "linear" 
    pool_mode = "none"  
    embedding_model = TimmWrapper(
        backbone_name="vit_giant_patch14_dinov2.lvd142m",
        embedding_size=embedding_size,
        embedding_id=embedding_id,
        dropout_p=dropout_p,
        pool_mode=pool_mode,
        img_size=224
    ).to(device)
    
    # Load the model weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # remove the prefix "module." from the keys of the state_dict
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    embedding_model.load_state_dict(state_dict)
    embedding_model = embedding_model.to(memory_format=torch.channels_last)
    print(f"Model device: {next(embedding_model.parameters()).device}")  # Debugging

    embedding_model.eval()
    return embedding_model

def get_dataloader(image_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts the image to a PyTorch tensor and scales pixel values to [0, 1].
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizes the tensor to have a mean of 0 and a standard deviation of 1.
    ])
    
    dataset = UnlabeledImageDataset(root_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=False)
    return dataloader, dataset

def extract_embeddings(model, data_loader, device, use_amp=True):
    model.to(device)
    
    embeddings_tensor = torch.tensor([], device=device)
    image_paths = []

    model.eval()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_amp):
        for images, paths in tqdm(data_loader, desc="Generating embeddings"):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            embeddings = model(images)
            embeddings_tensor = torch.cat([embeddings_tensor, embeddings])
            image_paths.extend(paths)
    
    embeddings = embeddings_tensor.cpu().numpy()
    return embeddings, image_paths

def main():
    parser = argparse.ArgumentParser(description="Clustering script for image embeddings")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading (default: 32)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97], help="List of distance thresholds for clustering")
    parser.add_argument("--constrained", action="store_true", help="Use constrained clustering (precomputed distance matrix)")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision (default: True)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir) or not os.listdir(args.image_dir):
        print(f"Error: image_dir '{args.image_dir}' does not exist or is empty.")
        return
    
    try:
        model = load_vit_model(args.checkpoint_path, DEVICE)
        print("Model loaded successfully.")
    except RuntimeError as e:
        print("Error loading model:", e)
        return
    
    dataloader, _ = get_dataloader(args.image_dir, args.batch_size)
    print("Images loaded successfully.")
    
    embeddings, image_paths = extract_embeddings(model, dataloader, DEVICE, use_amp=args.use_amp)
    print("Embeddings extracted successfully.")
    
    list_of_labels = []
    ground_truth_labels = [os.path.basename(path)[:4] for path in image_paths]
    
    label_to_int = {label: idx for idx, label in enumerate(sorted(set(ground_truth_labels)))}
    num_distinct_labels = len(label_to_int)
    ground_truth_labels = [label_to_int[label] for label in ground_truth_labels]
    
    ari_scores = []
    nmi_scores = []
    
    if args.constrained:
        start_time = time.time()
        distance_matrix = get_distance_matrix(image_paths, embeddings)
        print(distance_matrix)
        end_time = time.time()
        print(f"Distance matrix computed in {end_time - start_time:.2f} seconds")
        
    for threshold in args.thresholds:

        if args.constrained:
            clustering = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, metric="precomputed", linkage="complete")
            labels = clustering.fit_predict(distance_matrix)
        else:
            clustering = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, metric="euclidean", linkage="complete")
            labels = clustering.fit_predict(embeddings)
        list_of_labels.append(labels)   
        
        print(f"Clustering completed for threshold {threshold}.")

        num_clusters = len(set(labels))
        ari = adjusted_rand_score(ground_truth_labels, labels)
        nmi = normalized_mutual_info_score(ground_truth_labels, labels)
        
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        
        print(f"{num_clusters} (number of clusters), {num_distinct_labels} (expected)")
        print(f"{round(ari, 3)} (ARI)")
        print(f"{round(nmi, 3)} (NMI)")


    print(f"Maximum ARI score: {max(ari_scores)}")
    print(f"Maximum NMI score: {max(nmi_scores)}")

if __name__ == "__main__":
    main()