from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import RandAugment

from gorillawatch.data_hf.gorilla_dataset import GorillaDataset


DEFAULT_HF_DATASET_ID = "gorilla-watch/Gorilla-SPAC-Wild"
DEFAULT_HF_DATASET_CONFIG = "face"


def _resolve_hf_dataset_id(data_path, hf_dataset_id):
    """Resolve which HF dataset to use from parameters."""
    if hf_dataset_id:
        return hf_dataset_id
    
    if isinstance(data_path, str) and "/" in data_path and not data_path.startswith("/"):
        return data_path
    
    return DEFAULT_HF_DATASET_ID



def _collect_labels_videos_from_splits(split_map):
    all_labels = []
    all_videos = []
    all_dates = []
    all_cameras = []
    for split_dataset in split_map.values():
        all_labels.extend(str(label) for label in split_dataset["class"])
        all_videos.extend(str(video) for video in split_dataset["video"])
        if "date" in split_dataset.column_names:
            all_dates.extend(str(date) for date in split_dataset["date"])
        else:
            all_dates.extend(["unknown"] * len(split_dataset))
        if "camera" in split_dataset.column_names:
            all_cameras.extend(str(camera) for camera in split_dataset["camera"])
        else:
            all_cameras.extend(["unknown"] * len(split_dataset))
    return all_labels, all_videos, all_dates, all_cameras


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandAugment(num_ops=0, magnitude=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_image_paths(root_dir=None, split="train", dataset_id=DEFAULT_HF_DATASET_ID, dataset_config=DEFAULT_HF_DATASET_CONFIG):
    """Compatibility helper used by qualitative evaluation utilities."""
    del root_dir
    split_dataset = load_dataset(dataset_id, dataset_config, split=split)
    if "file_name" in split_dataset.column_names:
        return [str(name) for name in split_dataset["file_name"]]
    return [str(i) for i in range(len(split_dataset))]


def get_image_file_names(root_dir):
    """Legacy API alias to preserve existing imports."""
    del root_dir
    return get_image_paths()


def filter_image_names_by_labels(image_file_names, labels):
    """Legacy API for label filtering over image identifiers."""
    return [
        name for name in image_file_names
        if (label := str(name).split("/")[-1].split("_")[0]) in labels
    ]


def create_dataset_for_NON_SPAC(data_path, image_size):
    """Compatibility wrapper that now uses the Hugging Face train split."""
    transform = get_transform(image_size)
    dataset_id = data_path if isinstance(data_path, str) and "/" in data_path and not data_path.startswith("/") else DEFAULT_HF_DATASET_ID
    hf_train = load_dataset(dataset_id, DEFAULT_HF_DATASET_CONFIG, split="train")
    all_dataset = GorillaDataset(hf_train, transform=transform, split_name="train")
    return all_dataset


def prepare_data(
    data_path: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int,
    k: int,
    verbose: bool = True,
    parallelize: bool = True,
    hf_dataset_id: str = None,
    dataset_config: str = DEFAULT_HF_DATASET_CONFIG,
) -> dict:
    """
    Prepare data splits and dataloaders for training and evaluation.

    Returns:
        dict: Dictionary with split names as keys and (dataset, dataloader) tuples as values.
              Keys include: "train", "val", "test", "single_encounter"
    """
    random_seed = seed
    hf_dataset_id = _resolve_hf_dataset_id(data_path, hf_dataset_id)

    hf_dataset = load_dataset(hf_dataset_id, dataset_config)

    # Only use available splits from the dataset
    split_mapping = {
        "train": "train",
        "val": "validation",
        "test": "test",
        "single_encounter": "single_encounter"
    }

    split_files = {}
    for split_name, hf_split_name in split_mapping.items():
        if hf_split_name in hf_dataset:
            split_files[split_name] = hf_dataset[hf_split_name]


    all_labels, all_videos, all_dates, all_cameras = _collect_labels_videos_from_splits(split_files)

    label_to_idx = {label: i for i, label in enumerate(sorted(set(all_labels)))}
    video_to_idx = {video: i for i, video in enumerate(sorted(set(all_videos)))}
    date_to_idx = {date: i for i, date in enumerate(sorted(set(all_dates)))}
    camera_to_idx = {camera: i for i, camera in enumerate(sorted(set(all_cameras)))}

    if verbose:
        for split, split_dataset in split_files.items():
            print(f"Number of {split} images: {len(split_dataset)}")

    transform = get_transform(image_size)
    datasets = {
        split_name: GorillaDataset(
            hf_split,
            transform=transform,
            k=k,
            label_to_idx=label_to_idx,
            video_to_idx=video_to_idx,
            date_to_idx=date_to_idx,
            camera_to_idx=camera_to_idx,
            split_name=split_name,
        )
        for split_name, hf_split in split_files.items()
    }

    def create_data_loader_args(dataset, is_train):
        sampler = DistributedSampler(dataset, drop_last=True) if is_train and parallelize else None
        return {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": is_train and sampler is None,
            "num_workers": num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }

    loaders = {
        name: DataLoader(**create_data_loader_args(dataset, name == "train"))
        for name, dataset in datasets.items()
    }

    # Return as dictionary with split names as keys and (dataset, dataloader) tuples as values
    return {
        split_name: (datasets[split_name], loaders[split_name])
        for split_name in split_files.keys()
    }
