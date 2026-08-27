import json
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import RandAugment
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gorillawatch.data.gorilla_dataset import GorillaDataset, extract_labels_and_videos

def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandAugment(num_ops=0, magnitude=8), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_image_file_names(root_dir):
    return [
        name 
        for _, _, names in os.walk(root_dir) 
        for name in names 
        if name.endswith(('.png', '.jpg', '.jpeg'))
    ]

def filter_image_names_by_labels(image_file_names, labels):
    """Filters the image file names to include only those belonging to the specified labels."""
    return [
        name for name in image_file_names
        if (label := os.path.splitext(name)[0].split("_")[0]) in labels
    ]

def create_dataset_for_NON_SPAC(data_path, image_size):
    transform = get_transform(image_size)
    image_paths = get_image_file_names(data_path)
    all_dataset = GorillaDataset(data_path, image_paths, transform=transform)
    return all_dataset
    
def prepare_data(
    data_path: str,
    split_path: str,
    batch_size: int,
    num_workers: int,
    train_val_ratio: float,
    seed: int,
    image_size: int,
    k: int,
    wandb_run=None,
    verbose: bool = True,
    parallelize: bool = True,
    perform_train_val_split: bool = True,
) -> tuple:
    random_seed = seed
    
    file_names = get_image_file_names(data_path)
    all_labels, all_videos, all_dates, all_cameras = extract_labels_and_videos(file_names)

    # Encode labels, videos, dates, and cameras before data split
    label_to_idx = {label: i for i, label in enumerate(sorted(set(all_labels)))}
    video_to_idx = {video: i for i, video in enumerate(sorted(set(all_videos)))}
    date_to_idx = {date: i for i, date in enumerate(sorted(set(all_dates)))}
    camera_to_idx = {camera: i for i, camera in enumerate(sorted(set(all_cameras)))}
    
    splits = ['train', 'val', 'test']
    if split_path: 
        with open(split_path, 'r') as f: 
            split_labels = json.load(f) 
        split_files = {
            split: filter_image_names_by_labels(file_names, split_labels[renamed_split])
            for split in splits
            if (renamed_split := "validation" if split == "val" else split)
        }
    else: 
        split_files = {
            split: [os.path.join(split, img) for img in get_image_file_names(path)]
            for split in splits
            if (path := os.path.join(data_path, split)) and os.path.isdir(path)
        }
        
    if perform_train_val_split:
        print(f"Number of train images before split: {len(split_files['train'])}") 
        split_files['train'], split_files['train_val'] = train_test_split( 
            split_files['train'], test_size=train_val_ratio, 
            random_state=random_seed, shuffle=True 
        ) 
    for split, file_names in split_files.items():
        print(f"Number of {split} images: {len(file_names)}")

    transform = get_transform(image_size) 
    datasets = {
        split_name: GorillaDataset(data_path, files, transform=transform, k=k, label_to_idx=label_to_idx, video_to_idx=video_to_idx, date_to_idx=date_to_idx, camera_to_idx=camera_to_idx)
        for split_name, files in split_files.items()
    }

    def create_data_loader_args(dataset, is_train):
        return {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": is_train and parallelize,
            "num_workers": num_workers,
            "pin_memory": True,
            "sampler": DistributedSampler(dataset, drop_last=True) if is_train and parallelize else None
        }
    loaders = {
        name: DataLoader(**create_data_loader_args(dataset, name == 'train'))
        for name, dataset in datasets.items()
    }
    if "train_val" in loaders:
        return (
            datasets["train"], loaders["train"], 
            datasets["train_val"], loaders["train_val"],
            datasets["val"], loaders["val"], 
            datasets["test"], loaders["test"]
        )
    return (
        datasets["train"], loaders["train"], 
        datasets["val"], loaders["val"], 
        datasets["test"], loaders["test"]
    )
