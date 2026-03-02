"""
data/dataset.py - Dataset Loading and Preparation
===================================================
Reads image data from FER2013 folder structure, RAF-DB and CK+ datasets.
Creates PyTorch Dataset classes and DataLoaders.

FER2013 Folder Structure (Kaggle image format):
    fer2013/
    +-- train/
    |   +-- angry/
    |   +-- disgust/
    |   +-- fear/
    |   +-- happy/
    |   +-- sad/
    |   +-- surprise/
    +-- test/
        +-- angry/
        +-- ...

Each subfolder contains 48x48 grayscale JPG images.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Folder name to label mapping (6 Ekman + Neutral)
FOLDER_TO_LABEL = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}


class FER2013Dataset(Dataset):
    """
    PyTorch Dataset class for FER2013 (folder-based).

    Args:
        root_dir (str): Image root directory (e.g., data/fer2013/train)
        transform (callable, optional): Transforms to apply to images
        img_size (int): Target image size (default: 48)
        num_channels (int): Number of channels (1=grayscale, 3=RGB)
    """

    def __init__(self, root_dir, transform=None, img_size=None, num_channels=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size or config.IMG_SIZE
        self.num_channels = num_channels or config.NUM_CHANNELS

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"Data directory not found: {root_dir}\n"
                f"Please place FER2013 dataset under data/fer2013/."
            )

        # Scan each emotion folder
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Map folder name to label
            label_name = folder_name.lower()
            if label_name not in FOLDER_TO_LABEL:
                print(f"[WARNING] Unknown folder skipped: {folder_name}")
                continue

            label = FOLDER_TO_LABEL[label_name]

            # Collect all images in the folder
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for img_file in image_files:
                self.image_paths.append(os.path.join(folder_path, img_file))
                self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"[INFO] {root_dir}: {len(self.image_paths)} images loaded "
              f"({len(set(self.labels))} classes)")

    def __len__(self):
        """Total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the image and label at the specified index.

        Returns:
            image (Tensor): Normalized image of shape [1, 48, 48]
            label (int): Emotion label in range 0-5 (6 Ekman)
        """
        # Read image
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Convert to target color mode
        if self.num_channels == 3:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            if image.mode != 'L':
                image = image.convert('L')

        # Resize to target size
        if image.size != (self.img_size, self.img_size):
            image = image.resize(
                (self.img_size, self.img_size), Image.LANCZOS
            )

        # Convert to numpy array and normalize [0, 255] -> [0.0, 1.0]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Add channel dimension
        if self.num_channels == 1:
            img_array = np.expand_dims(img_array, axis=0)  # (1, H, W)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))  # (3, H, W)

        # Convert to tensor
        tensor = torch.FloatTensor(img_array)

        # Apply augmentation if available
        if self.transform:
            tensor = self.transform(tensor)

        label = torch.LongTensor([self.labels[idx]]).squeeze()

        return tensor, label

    def get_sample_images(self, num_per_class=3):
        """
        Returns sample images from each class as NumPy arrays.

        Returns:
            dict: {label_id: [numpy_array, ...], ...}
        """
        samples = {}
        for cls_id in range(config.NUM_CLASSES):
            class_indices = np.where(self.labels == cls_id)[0]
            if len(class_indices) == 0:
                continue

            selected = np.random.choice(
                class_indices,
                min(num_per_class, len(class_indices)),
                replace=False
            )

            images = []
            for idx in selected:
                img = Image.open(self.image_paths[idx]).convert('L')
                images.append(np.array(img))

            samples[cls_id] = images

        return samples


def get_transforms(split="train", model_name="mini_xception"):
    """
    Returns appropriate data transforms for the given split.

    For transfer learning models (efficientnet, resnet, hsemotion),
    adds ImageNet normalization.

    Args:
        split (str): 'train' or 'test'
        model_name (str): Model name for model-specific transforms

    Returns:
        transforms.Compose or None
    """
    is_transfer = model_name in ("efficientnet", "resnet", "hsemotion")

    if split == "train":
        aug_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=config.AUGMENTATION["rotation_degrees"]
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=config.AUGMENTATION["random_affine_translate"]
            ),
            transforms.RandomErasing(
                p=config.AUGMENTATION["random_erasing_prob"],
                scale=(0.02, 0.1)
            ),
        ]
        # Add ImageNet normalization for transfer learning models
        if is_transfer:
            aug_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        return transforms.Compose(aug_list)
    else:
        # Test/Val: only normalization for transfer learning
        if is_transfer:
            return transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        return None


def get_dataloaders(data_dir=None, batch_size=None, val_split=0.15,
                    model_name="mini_xception"):
    """
    Creates Train, Validation, and Test DataLoaders.

    FER2013 folder structure:
    - train/  ->  Training + Validation (85% / 15% random split)
    - test/   ->  Test set

    Args:
        data_dir (str): FER2013 root directory (default: from config)
        batch_size (int): Batch size
        val_split (float): Fraction of training set for validation
        model_name (str): Model name for model-specific settings

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    if data_dir is None:
        data_dir = config.FER2013_DIR

    # Get model-specific settings
    model_cfg = config.MODEL_CONFIGS.get(model_name, config.MODEL_CONFIGS["mini_xception"])
    if batch_size is None:
        batch_size = model_cfg["batch_size"]
    img_size = model_cfg["img_size"]
    num_channels = model_cfg["num_channels"]

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    print("\n" + "=" * 60)
    print(f"  Loading FER2013 Dataset (img={img_size}x{img_size}, ch={num_channels})")
    print("=" * 60)

    # Training set -- without augmentation initially
    full_train_dataset = FER2013Dataset(
        train_dir, transform=None,
        img_size=img_size, num_channels=num_channels
    )

    # Split training set into Train + Validation
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducibility
    )

    # Augmentation: applied only to training subset
    train_dataset = TransformSubset(train_subset, get_transforms("train", model_name))
    val_dataset = TransformSubset(val_subset, get_transforms("test", model_name))

    # Test set -- no augmentation
    test_dataset = FER2013Dataset(
        test_dir, transform=get_transforms("test", model_name),
        img_size=img_size, num_channels=num_channels
    )

    # DataLoaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,       # Windows compatibility
            pin_memory=torch.cuda.is_available()
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        ),
    }

    # Summary
    print(f"\n[INFO] Dataset sizes:")
    print(f"  Train:      {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    print(f"  Test:       {len(test_dataset):,} samples")
    print(f"  Batch:      {batch_size}")
    print(f"  Device:     {config.DEVICE}")
    print("=" * 60 + "\n")

    return dataloaders


class TransformSubset:
    """
    Wrapper to apply transforms after random_split.
    Necessary because PyTorch's Subset doesn't support transforms directly.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_class_distribution(data_dir=None):
    """
    Computes class distribution for each split.

    Returns:
        dict: {'train': {0: count, ...}, 'test': {0: count, ...}}
    """
    if data_dir is None:
        data_dir = config.FER2013_DIR

    distribution = {}

    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"[WARNING] {split_dir} not found, skipping.")
            continue

        counts = {}
        for folder_name in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            label_name = folder_name.lower()
            if label_name not in FOLDER_TO_LABEL:
                continue

            label_id = FOLDER_TO_LABEL[label_name]
            num_images = len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            counts[label_id] = num_images

        distribution[split] = counts

    # Print distribution
    print("\n" + "=" * 60)
    print("  Class Distribution")
    print("=" * 60)

    splits = list(distribution.keys())
    header = f"{'Class':<12}" + "".join(f"{s:>10}" for s in splits)
    print(f"\n{header}")
    print("-" * (12 + 10 * len(splits)))

    for cls_id in range(config.NUM_CLASSES):
        name = config.EMOTION_LABELS[cls_id]
        row = f"{name:<12}"
        for split in splits:
            count = distribution[split].get(cls_id, 0)
            row += f"{count:>10,}"
        print(row)

    print("-" * (12 + 10 * len(splits)))

    # Total
    row = f"{'TOTAL':<12}"
    for split in splits:
        total = sum(distribution[split].values())
        row += f"{total:>10,}"
    print(row)
    print("=" * 60 + "\n")

    return distribution


def get_class_weights(data_dir=None):
    """
    Computes class weights to handle imbalanced class distribution.
    Minority classes (e.g., Disgust) receive higher weights.

    Returns:
        torch.Tensor: Weight tensor for each class
    """
    if data_dir is None:
        data_dir = config.FER2013_DIR

    train_dir = os.path.join(data_dir, "train")
    counts = {}

    for folder_name in sorted(os.listdir(train_dir)):
        folder_path = os.path.join(train_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label_name = folder_name.lower()
        if label_name not in FOLDER_TO_LABEL:
            continue

        label_id = FOLDER_TO_LABEL[label_name]
        num_images = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        counts[label_id] = num_images

    total = sum(counts.values())
    num_classes_present = len(counts)

    # Inverse frequency weighting
    weights = []
    for i in range(config.NUM_CLASSES):
        if i in counts and counts[i] > 0:
            w = total / (num_classes_present * counts[i])
        else:
            w = 1.0  # Default weight if class is missing
        weights.append(w)

    weights_tensor = torch.FloatTensor(weights)

    print("[INFO] Class weights:")
    for i, w in enumerate(weights):
        count = counts.get(i, 0)
        print(f"  {config.EMOTION_LABELS[i]:<12}: {w:.4f}  ({count:,} samples)")

    return weights_tensor


# ================================================================
# RAF-DB Dataset
# ================================================================
class RAFDBDataset(Dataset):
    """
    PyTorch Dataset class for RAF-DB.

    RAF-DB structure:
        raf-db/
        +-- DATASET/
        |   +-- train/
        |   |   +-- 1/   (Surprise)
        |   |   +-- 2/   (Fear)
        |   |   +-- ...
        |   |   +-- 7/   (Neutral) -> filtered out
        |   +-- test/
        |       +-- 1/ ... 7/
        +-- train_labels.csv
        +-- test_labels.csv

    Images: 100x100 RGB -> converted to 48x48 grayscale.
    Label mapping: RAF-DB (1-7) -> standard (0-5), Neutral excluded.
    """

    def __init__(self, root_dir, split="train", transform=None,
                 img_size=None, num_channels=None):
        """
        Args:
            root_dir (str): RAF-DB root directory (data/raf-db)
            split (str): 'train' or 'test'
            transform (callable): Image transforms
            img_size (int): Target image size
            num_channels (int): Number of channels (1=grayscale, 3=RGB)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size or config.IMG_SIZE
        self.num_channels = num_channels or config.NUM_CHANNELS
        self.image_paths = []
        self.labels = []

        img_dir = os.path.join(root_dir, "DATASET", split)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(
                f"RAF-DB directory not found: {img_dir}\n"
                f"Please place RAF-DB dataset under data/raf-db/."
            )

        # Folder-based reading (numbered folders 1-7)
        for class_folder in sorted(os.listdir(img_dir)):
            class_path = os.path.join(img_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            try:
                raf_label = int(class_folder)
            except ValueError:
                continue

            if raf_label not in config.RAFDB_LABEL_MAP:
                # Neutral (label 7) is skipped here
                continue

            fer_label = config.RAFDB_LABEL_MAP[raf_label]

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(
                        os.path.join(class_path, img_file)
                    )
                    self.labels.append(fer_label)

        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"[INFO] RAF-DB {split}: {len(self.image_paths)} images loaded "
              f"({len(set(self.labels))} classes)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns image tensor with configurable size and channels."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Convert to target color mode
        if self.num_channels == 3:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            if image.mode != 'L':
                image = image.convert('L')

        # Resize to target size
        if image.size != (self.img_size, self.img_size):
            image = image.resize(
                (self.img_size, self.img_size), Image.LANCZOS
            )

        # Normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        if self.num_channels == 1:
            img_array = np.expand_dims(img_array, axis=0)  # (1, H, W)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))  # (3, H, W)

        tensor = torch.FloatTensor(img_array)

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return tensor, label


# ================================================================
# CK+ (Extended Cohn-Kanade) Dataset
# ================================================================
class CKPlusDataset(Dataset):
    """
    PyTorch Dataset class for CK+.

    CSV format (FER2013-like):
        emotion,pixels,Usage
        6, 36 39 35 ..., Training

    - Contempt (label=1) and Neutral (label=6) are removed.
    - CK+ labels are mapped to 6 Ekman labels.
    - Pixel data is 48x48 grayscale.
    """

    def __init__(self, csv_path, split="train", transform=None,
                 img_size=None, num_channels=None):
        """
        Args:
            csv_path (str): CK+ CSV file path
            split (str): 'train' or 'test' (filtered by Usage column)
            transform (callable): Image transforms
            img_size (int): Target image size
            num_channels (int): Number of channels
        """
        import pandas as pd

        self.transform = transform
        self.img_size = img_size or config.IMG_SIZE
        self.num_channels = num_channels or config.NUM_CHANNELS
        self.images = []
        self.labels = []

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CK+ CSV not found: {csv_path}\n"
                f"Please place ckextended.csv under data/ck+/."
            )

        df = pd.read_csv(csv_path)

        # Filter by Usage column
        usage_map = {"train": "Training", "test": "Testing",
                      "val": "PublicTest"}
        usage_filter = usage_map.get(split, "Training")

        if "Usage" in df.columns:
            df = df[df["Usage"] == usage_filter]

        skipped = 0

        for _, row in df.iterrows():
            ck_emotion = int(row["emotion"])

            # Skip Contempt and Neutral
            if ck_emotion not in config.CKPLUS_TO_FER_MAP:
                skipped += 1
                continue

            fer_label = config.CKPLUS_TO_FER_MAP[ck_emotion]

            # Convert pixel string to 48x48 numpy array
            pixels = np.array(
                row["pixels"].split(), dtype=np.float32
            ).reshape(config.IMG_SIZE, config.IMG_SIZE)

            self.images.append(pixels)
            self.labels.append(fer_label)

        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        info = f"[INFO] CK+ {split}: {len(self.images)} images loaded "
        info += f"({len(set(self.labels))} classes)"
        if skipped > 0:
            info += f" [{skipped} Contempt/Neutral samples skipped]"
        print(info)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Returns image tensor with configurable size and channels."""
        image = self.images[idx] / 255.0  # Normalize

        # Resize if needed
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            pil_img = pil_img.resize((self.img_size, self.img_size), Image.LANCZOS)
            image = np.array(pil_img, dtype=np.float32) / 255.0

        if self.num_channels == 3:
            # Grayscale -> RGB by repeating channels
            image = np.stack([image] * 3, axis=0)  # (3, H, W)
        else:
            image = np.expand_dims(image, axis=0)  # (1, H, W)

        tensor = torch.FloatTensor(image)

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return tensor, label


# ================================================================
# FERPlus Dataset (Folder-based, like FER2013 but with different labels)
# ================================================================
class FERPlusDataset(Dataset):
    """
    PyTorch Dataset class for FER+ (folder-based).

    FER+ structure:
        ferplus/
        +-- train/
        |   +-- angry/  disgust/  fear/  happy/  sad/  suprise/
        |   +-- contempt/  neutral/  (filtered out)
        +-- validation/
        +-- test/

    Labels: contempt and neutral are excluded for 6 Ekman emotions.
    Note: 'suprise' folder is misspelled in the dataset.
    """

    def __init__(self, root_dir, transform=None, img_size=None, num_channels=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size or config.IMG_SIZE
        self.num_channels = num_channels or config.NUM_CHANNELS
        self.image_paths = []
        self.labels = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"FER+ directory not found: {root_dir}\n"
                f"Please place FER+ dataset under data/ferplus/."
            )

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            label_name = folder_name.lower()
            if label_name not in config.FERPLUS_FOLDER_TO_LABEL:
                continue  # Skip contempt, neutral

            label = config.FERPLUS_FOLDER_TO_LABEL[label_name]

            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for img_file in image_files:
                self.image_paths.append(os.path.join(folder_path, img_file))
                self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"[INFO] {root_dir}: {len(self.image_paths)} images loaded "
              f"({len(set(self.labels))} classes)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.num_channels == 3:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            if image.mode != 'L':
                image = image.convert('L')

        if image.size != (self.img_size, self.img_size):
            image = image.resize(
                (self.img_size, self.img_size), Image.LANCZOS
            )

        img_array = np.array(image, dtype=np.float32) / 255.0
        if self.num_channels == 1:
            img_array = np.expand_dims(img_array, axis=0)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))

        tensor = torch.FloatTensor(img_array)
        if self.transform:
            tensor = self.transform(tensor)

        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return tensor, label


# ================================================================
# Factory Function -- Create DataLoaders by dataset name
# ================================================================
def get_dataloaders_for_dataset(dataset_name, batch_size=None, val_split=0.15,
                                model_name="mini_xception"):
    """
    Creates Train/Val/Test DataLoaders for the specified dataset.

    Args:
        dataset_name (str): 'fer2013', 'ferplus', 'rafdb', or 'ckplus'
        batch_size (int): Batch size
        val_split (float): Fraction of training set for validation
        model_name (str): Model name for model-specific settings

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    # Get model-specific batch size if not specified
    if batch_size is None:
        model_cfg = config.MODEL_CONFIGS.get(model_name, config.MODEL_CONFIGS["mini_xception"])
        batch_size = model_cfg["batch_size"]

    dataset_name = dataset_name.lower()

    if dataset_name == "fer2013":
        return get_dataloaders(batch_size=batch_size, val_split=val_split,
                              model_name=model_name)

    elif dataset_name == "ferplus":
        return _get_ferplus_dataloaders(batch_size, model_name)

    elif dataset_name == "rafdb":
        return _get_rafdb_dataloaders(batch_size, val_split, model_name)

    elif dataset_name == "ckplus":
        return _get_ckplus_dataloaders(batch_size, val_split, model_name)

    else:
        available = ", ".join(config.DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )


def _get_ferplus_dataloaders(batch_size, model_name="mini_xception"):
    """Creates FER+ DataLoaders using pre-existing train/validation/test splits."""
    model_cfg = config.MODEL_CONFIGS.get(model_name, config.MODEL_CONFIGS["mini_xception"])
    img_size = model_cfg["img_size"]
    num_channels = model_cfg["num_channels"]

    print("\n" + "=" * 60)
    print(f"  Loading FER+ Dataset (img={img_size}x{img_size}, ch={num_channels})")
    print("=" * 60)

    # FER+ has pre-existing train/validation/test splits
    train_dir = os.path.join(config.FERPLUS_DIR, "train")
    val_dir = os.path.join(config.FERPLUS_DIR, "validation")
    test_dir = os.path.join(config.FERPLUS_DIR, "test")

    train_dataset = FERPlusDataset(
        train_dir, transform=get_transforms("train", model_name),
        img_size=img_size, num_channels=num_channels
    )
    val_dataset = FERPlusDataset(
        val_dir, transform=get_transforms("test", model_name),
        img_size=img_size, num_channels=num_channels
    )
    test_dataset = FERPlusDataset(
        test_dir, transform=get_transforms("test", model_name),
        img_size=img_size, num_channels=num_channels
    )

    dataloaders = _build_dataloaders(train_dataset, val_dataset,
                                      test_dataset, batch_size)

    _print_summary("FER+", train_dataset, val_dataset,
                    test_dataset, batch_size)
    return dataloaders


def _get_rafdb_dataloaders(batch_size, val_split, model_name="mini_xception"):
    """Creates RAF-DB DataLoaders."""
    model_cfg = config.MODEL_CONFIGS.get(model_name, config.MODEL_CONFIGS["mini_xception"])
    img_size = model_cfg["img_size"]
    num_channels = model_cfg["num_channels"]

    print("\n" + "=" * 60)
    print(f"  Loading RAF-DB Dataset (img={img_size}x{img_size}, ch={num_channels})")
    print("=" * 60)

    # Training set
    full_train = RAFDBDataset(config.RAFDB_DIR, split="train",
                              img_size=img_size, num_channels=num_channels)

    # Train -> Train + Validation split
    total = len(full_train)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_subset, val_subset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = TransformSubset(train_subset, get_transforms("train", model_name))
    val_dataset = TransformSubset(val_subset, get_transforms("test", model_name))

    # Test set
    test_dataset = RAFDBDataset(config.RAFDB_DIR, split="test",
                                transform=get_transforms("test", model_name),
                                img_size=img_size, num_channels=num_channels)

    dataloaders = _build_dataloaders(train_dataset, val_dataset,
                                      test_dataset, batch_size)

    _print_summary("RAF-DB", train_dataset, val_dataset,
                    test_dataset, batch_size)
    return dataloaders


def _get_ckplus_dataloaders(batch_size, val_split, model_name="mini_xception"):
    """Creates CK+ DataLoaders."""
    model_cfg = config.MODEL_CONFIGS.get(model_name, config.MODEL_CONFIGS["mini_xception"])
    img_size = model_cfg["img_size"]
    num_channels = model_cfg["num_channels"]

    print("\n" + "=" * 60)
    print(f"  Loading CK+ Dataset (img={img_size}x{img_size}, ch={num_channels})")
    print("=" * 60)

    csv_path = os.path.join(config.CKPLUS_DIR, "ckextended.csv")

    # CK+ usually only has Training, so we manually split
    full_dataset = CKPlusDataset(csv_path, split="train",
                                 img_size=img_size, num_channels=num_channels)

    total = len(full_dataset)
    test_size = int(total * 0.15)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = TransformSubset(train_subset, get_transforms("train", model_name))
    val_dataset = TransformSubset(val_subset, get_transforms("test", model_name))
    test_dataset = TransformSubset(test_subset, get_transforms("test", model_name))

    dataloaders = _build_dataloaders(train_dataset, val_dataset,
                                      test_dataset, batch_size)

    _print_summary("CK+", train_dataset, val_dataset,
                    test_dataset, batch_size)
    return dataloaders


def _build_dataloaders(train_ds, val_ds, test_ds, batch_size):
    """Common DataLoader builder helper."""
    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        ),
    }


def _print_summary(name, train_ds, val_ds, test_ds, batch_size):
    """Common dataset summary printer."""
    print(f"\n[INFO] {name} Dataset sizes:")
    print(f"  Train:      {len(train_ds):,} samples")
    print(f"  Validation: {len(val_ds):,} samples")
    print(f"  Test:       {len(test_ds):,} samples")
    print(f"  Batch:      {batch_size}")
    print(f"  Device:     {config.DEVICE}")
    print("=" * 60 + "\n")


def get_class_weights_for_dataset(dataset_name, data_dir=None):
    """
    Computes class weights for any dataset.

    Args:
        dataset_name (str): 'fer2013', 'rafdb', 'ckplus'

    Returns:
        torch.Tensor: Class weights
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "fer2013":
        return get_class_weights(data_dir)

    elif dataset_name == "rafdb":
        train_dir = os.path.join(config.RAFDB_DIR, "DATASET", "train")
        counts = {}

        for class_folder in sorted(os.listdir(train_dir)):
            class_path = os.path.join(train_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            try:
                raf_label = int(class_folder)
            except ValueError:
                continue
            if raf_label in config.RAFDB_LABEL_MAP:
                fer_label = config.RAFDB_LABEL_MAP[raf_label]
                counts[fer_label] = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])

        return _compute_weights(counts)

    elif dataset_name == "ckplus":
        import pandas as pd
        csv_path = os.path.join(config.CKPLUS_DIR, "ckextended.csv")
        df = pd.read_csv(csv_path)
        counts = {}
        for _, row in df.iterrows():
            ck_em = int(row["emotion"])
            if ck_em in config.CKPLUS_TO_FER_MAP:
                fer_label = config.CKPLUS_TO_FER_MAP[ck_em]
                counts[fer_label] = counts.get(fer_label, 0) + 1
        return _compute_weights(counts)

    elif dataset_name == "ferplus":
        train_dir = os.path.join(config.FERPLUS_DIR, "train")
        counts = {}
        for folder_name in sorted(os.listdir(train_dir)):
            folder_path = os.path.join(train_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label_name = folder_name.lower()
            if label_name not in config.FERPLUS_FOLDER_TO_LABEL:
                continue
            fer_label = config.FERPLUS_FOLDER_TO_LABEL[label_name]
            counts[fer_label] = counts.get(fer_label, 0) + len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        return _compute_weights(counts)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _compute_weights(counts):
    """Compute and print class weights."""
    total = sum(counts.values())
    num_classes_present = len(counts)

    weights = []
    for i in range(config.NUM_CLASSES):
        if i in counts and counts[i] > 0:
            w = total / (num_classes_present * counts[i])
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.FloatTensor(weights)

    print("[INFO] Class weights:")
    for i, w in enumerate(weights):
        count = counts.get(i, 0)
        print(f"  {config.EMOTION_LABELS[i]:<12}: {w:.4f}  "
              f"({count:,} samples)")

    return weights_tensor
