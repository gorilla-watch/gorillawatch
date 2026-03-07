from collections import defaultdict
import os
from torch.utils.data import Dataset
from PIL import Image

def extract_labels_and_videos(file_list):
    """Extract (labels, videos) from a list of file paths."""
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    parts_list = [name.split("_") for name in file_names]
    labels = [parts[0] for parts in parts_list]
    videos = ["_".join(parts[1:4]) for parts in parts_list]
    return labels, videos

class GorillaDataset(Dataset):
    """
    Custom Dataset class for gorilla images with cross-video KNN support.
    Loads and processes image data, maintaining label and video information from filenames.
    Features:
    - Converts string labels and video IDs to numeric indices
    - Filters images for cross-video KNN based on minimum sample requirements
    - Organizes data by label and video for efficient KNN processing

    Parameters:
        root_dir (str): Directory containing the image files
        file_list (list): List of image filenames
        transform (callable, optional): Optional transform to be applied on images
        images_for_cv_knn (list): List of indices for images qualified as queries for cross-video KNN
        images_for_standard_knn (list): List of indices for images qualified as queries for standard KNN
    """
    def __init__(self, root_dir, file_list, transform=None, k=5,
                 label_to_idx=None, video_to_idx=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform
        self.k = k

        self.images = [os.path.join(root_dir, f) for f in file_list]
        self.labels, self.videos = extract_labels_and_videos(self.images)

        distinct_encounters = set(f"{label}, {video}" for label, video in zip(self.labels, self.videos))
        print(f"Found {len(distinct_encounters)} distinct encounters in dataset.")

        if label_to_idx is None:
            label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        if video_to_idx is None:
            video_to_idx = {video: idx for idx, video in enumerate(sorted(set(self.videos)))}

        self.label_to_idx = label_to_idx
        self.video_to_idx = video_to_idx
        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        self.idx_to_video = {idx: video for video, idx in video_to_idx.items()}

        self.labels = [self.label_to_idx[label] for label in self.labels]
        self.videos = [self.video_to_idx[video] for video in self.videos]

        # Organize for KNN filtering
        data_by_label = defaultdict(lambda: {"images": [], "videos": defaultdict(list)})
        for image, label, video in zip(self.images, self.labels, self.videos):
            data_by_label[label]["images"].append(image)
            data_by_label[label]["videos"][video].append(image)

        self.images_for_cv_knn = []
        self.images_for_standard_knn = []

        for idx, (image, label, video) in enumerate(zip(self.images, self.labels, self.videos)):
            videos_with_label = data_by_label[label]["videos"]
            other_videos_count = sum(len(imgs) for vid, imgs in videos_with_label.items() if vid != video)
            if other_videos_count >= self.k // 2 + 1:
                self.images_for_cv_knn.append(idx)
            if len(data_by_label[label]["images"]) >= self.k // 2 + 2:
                self.images_for_standard_knn.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        video = self.videos[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, video
