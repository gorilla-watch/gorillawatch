from collections import defaultdict
from io import BytesIO
from os import path

from PIL import Image
from torch.utils.data import Dataset


def extract_labels_and_videos(records):
    """Extract (labels, videos) from HF records or legacy filename lists."""
    if not records:
        return [], []

    first = records[0]
    if isinstance(first, dict):
        labels = [str(record["class"]) for record in records]
        videos = [str(record["video"]) for record in records]
        return labels, videos

    file_names = [path.splitext(path.basename(record))[0] for record in records]
    parts_list = [name.split("_") for name in file_names]
    labels = [parts[0] for parts in parts_list]
    videos = ["_".join(parts[1:4]) for parts in parts_list]
    return labels, videos


def _to_pil_rgb(image_obj):
    """Normalize datasets.Image output to RGB PIL.Image."""
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")

    if isinstance(image_obj, dict) and "bytes" in image_obj and image_obj["bytes"] is not None:
        return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")

    raise TypeError(f"Unsupported image type from HF dataset: {type(image_obj)}")


class GorillaDataset(Dataset):
    """Hugging Face-backed Gorilla dataset with cross-video KNN indexing support."""

    def __init__(
        self,
        hf_split,
        transform=None,
        k=5,
        label_to_idx=None,
        video_to_idx=None,
        date_to_idx=None,
        camera_to_idx=None,
        split_name=None,
    ):
        self.dataset = hf_split
        self.transform = transform
        self.k = k
        self.split_name = split_name

        self.raw_labels = [str(label) for label in hf_split["class"]]
        self.raw_videos = [str(video) for video in hf_split["video"]]
        self.raw_dates = [str(date) for date in hf_split["date"]] if "date" in hf_split.column_names else ["unknown"] * len(hf_split)
        self.raw_cameras = [str(cam) for cam in hf_split["camera"]] if "camera" in hf_split.column_names else ["unknown"] * len(hf_split)
        self.file_names = [str(name) for name in hf_split["file_name"]] if "file_name" in hf_split.column_names else []

        distinct_encounters = set(zip(self.raw_labels, self.raw_videos))
        print(f"Found {len(distinct_encounters)} distinct encounters in dataset.")

        if label_to_idx is None:
            label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.raw_labels)))}
        if video_to_idx is None:
            video_to_idx = {video: idx for idx, video in enumerate(sorted(set(self.raw_videos)))}
        if date_to_idx is None:
            date_to_idx = {date: idx for idx, date in enumerate(sorted(set(self.raw_dates)))}
        if camera_to_idx is None:
            camera_to_idx = {camera: idx for idx, camera in enumerate(sorted(set(self.raw_cameras)))}

        self.label_to_idx = label_to_idx
        self.video_to_idx = video_to_idx
        self.date_to_idx = date_to_idx
        self.camera_to_idx = camera_to_idx
        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        self.idx_to_video = {idx: video for video, idx in video_to_idx.items()}
        self.idx_to_date = {idx: date for date, idx in date_to_idx.items()}
        self.idx_to_camera = {idx: camera for camera, idx in camera_to_idx.items()}

        self.labels = [self.label_to_idx[label] for label in self.raw_labels]
        self.videos = [self.video_to_idx[video] for video in self.raw_videos]
        self.dates = [self.date_to_idx[date] for date in self.raw_dates]
        self.cameras = [self.camera_to_idx[camera] for camera in self.raw_cameras]

        self.images = self.file_names if self.file_names else [str(i) for i in range(len(self.labels))]

        data_by_label = defaultdict(lambda: {"count": 0, "videos": defaultdict(int)})
        for label, video in zip(self.labels, self.videos):
            data_by_label[label]["count"] += 1
            data_by_label[label]["videos"][video] += 1

        self.images_for_cv_knn = []
        self.images_for_standard_knn = []

        for idx, (label, video) in enumerate(zip(self.labels, self.videos)):
            videos_with_label = data_by_label[label]["videos"]
            other_videos_count = sum(count for vid, count in videos_with_label.items() if vid != video)
            if other_videos_count >= self.k // 2 + 1:
                self.images_for_cv_knn.append(idx)
            if data_by_label[label]["count"] >= self.k // 2 + 2:
                self.images_for_standard_knn.append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = _to_pil_rgb(sample["image"])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.videos[idx], self.dates[idx], self.cameras[idx]
