"""
config.py - Project Configuration
==================================
All hyperparameters, paths, and constants are defined here.
Easily extensible for different datasets.
"""

import os
import torch

# ============================================================
# Project Directory Paths
# ============================================================
# Root directory = directory containing this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FER2013_DIR = os.path.join(DATA_DIR, "fer2013")  # Folder-based dataset
FERPLUS_DIR = os.path.join(DATA_DIR, "ferplus")  # FER+ dataset (folder-based)
RAFDB_DIR = os.path.join(DATA_DIR, "raf-db")     # RAF-DB dataset
CKPLUS_DIR = os.path.join(DATA_DIR, "ck+")       # CK+ dataset

# Output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# Device Settings
# ============================================================
# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Emotion Labels (6 Ekman Emotions)
# ============================================================
# 6 basic Ekman emotions + Neutral
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

NUM_CLASSES = len(EMOTION_LABELS)  # 7
IMG_SIZE = 48   # FER2013 image size: 48x48 pixels
NUM_CHANNELS = 1  # Grayscale

# RAF-DB labels (original 1-7 mapped to standard 0-6)
RAFDB_LABEL_MAP = {
    1: 5,  # Surprise  -> 5
    2: 2,  # Fear      -> 2
    3: 1,  # Disgust   -> 1
    4: 3,  # Happiness -> 3
    5: 4,  # Sadness   -> 4
    6: 0,  # Anger     -> 0
    7: 6,  # Neutral   -> 6
}

# CK+ labels (8 classes -- Contempt is an extra class)
CKPLUS_EMOTION_LABELS = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Happy",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral",
    7: "Fear"
}

# CK+ -> 7 class mapping (Contempt removed, Neutral included)
CKPLUS_TO_FER_MAP = {
    0: 0,  # Anger    -> Angry
    2: 1,  # Disgust  -> Disgust
    7: 2,  # Fear     -> Fear
    3: 3,  # Happy    -> Happy
    4: 4,  # Sadness  -> Sad
    5: 5,  # Surprise -> Surprise
    6: 6,  # Neutral  -> Neutral
    # 1: Contempt -> removed
}

# FERPlus folder name -> 6 Ekman label mapping
# Note: 'suprise' is the actual folder name in the dataset (misspelled)
FERPLUS_FOLDER_TO_LABEL = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "suprise": 5,    # Misspelled in dataset
    "surprise": 5,   # Fallback if corrected
    "neutral": 6,
    # "contempt" -> removed
}

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 64       # Suitable for i5 + 4GB RAM
EPOCHS = 10           # Maximum number of epochs (CPU-friendly)
LEARNING_RATE = 1e-3  # Initial learning rate for Adam optimizer
WEIGHT_DECAY = 1e-4   # L2 regularization

# Learning Rate Scheduler
LR_PATIENCE = 5       # Reduce LR after this many epochs without improvement
LR_FACTOR = 0.5       # Multiply LR by this factor

# Early Stopping
EARLY_STOPPING_PATIENCE = 10  # Stop training after this many epochs without improvement

# ============================================================
# Data Augmentation
# ============================================================
AUGMENTATION = {
    "horizontal_flip": True,
    "rotation_degrees": 10,
    "random_affine_translate": (0.1, 0.1),
    "random_erasing_prob": 0.2
}

# ============================================================
# Model Selection
# ============================================================
# Available models: "mini_xception", "efficientnet", "resnet", "hsemotion"
MODEL_NAME = "mini_xception"  # Default model

# Transfer Learning Settings (shared)
TRANSFER_IMG_SIZE = 224       # Input size for pretrained models (EfficientNet, ResNet)
TRANSFER_NUM_CHANNELS = 3     # RGB input for pretrained models
TRANSFER_LR = 1e-4            # Lower LR for transfer learning
TRANSFER_BATCH_SIZE = 32      # Smaller batch for larger images

# EfficientNet Transfer Learning Settings
EFFICIENTNET_FREEZE_BACKBONE = True    # Freeze backbone layers
EFFICIENTNET_UNFREEZE_LAST_N = 2       # Unfreeze last N blocks
EFFICIENTNET_LR = 1e-4                 # Lower LR for transfer learning

# ResNet-18 Transfer Learning Settings
RESNET_FREEZE_BACKBONE = True          # Freeze backbone layers
RESNET_UNFREEZE_LAST_N = 2             # Unfreeze last N residual layers
RESNET_LR = 1e-4                       # Lower LR for transfer learning

# HSEmotion Settings
HSEMOTION_MODEL_NAME = "enet_b0_8_best_afew"  # Pre-trained model name

# Model-specific configurations
MODEL_CONFIGS = {
    "mini_xception": {
        "img_size": IMG_SIZE,          # 48
        "num_channels": NUM_CHANNELS,  # 1 (grayscale)
        "batch_size": BATCH_SIZE,      # 64
        "lr": LEARNING_RATE,           # 1e-3
    },
    "efficientnet": {
        "img_size": TRANSFER_IMG_SIZE,      # 224
        "num_channels": TRANSFER_NUM_CHANNELS, # 3 (RGB)
        "batch_size": TRANSFER_BATCH_SIZE,  # 32
        "lr": EFFICIENTNET_LR,              # 1e-4
    },
    "resnet": {
        "img_size": TRANSFER_IMG_SIZE,      # 224
        "num_channels": TRANSFER_NUM_CHANNELS, # 3 (RGB)
        "batch_size": TRANSFER_BATCH_SIZE,  # 32
        "lr": RESNET_LR,                    # 1e-4
    },
    "hsemotion": {
        "img_size": 260,                    # HSEmotion default input
        "num_channels": 3,                  # RGB
        "batch_size": TRANSFER_BATCH_SIZE,  # 32
        "lr": 5e-5,                         # Very low LR for fine-tuning
    },
}

# ============================================================
# Model Saving
# ============================================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pth")

# Model-specific paths
BEST_MODEL_PATHS = {
    "mini_xception": os.path.join(MODEL_DIR, "best_mini_xception.pth"),
    "efficientnet": os.path.join(MODEL_DIR, "best_efficientnet.pth"),
    "resnet": os.path.join(MODEL_DIR, "best_resnet.pth"),
    "hsemotion": os.path.join(MODEL_DIR, "best_hsemotion.pth"),
}
LAST_MODEL_PATHS = {
    "mini_xception": os.path.join(MODEL_DIR, "last_mini_xception.pth"),
    "efficientnet": os.path.join(MODEL_DIR, "last_efficientnet.pth"),
    "resnet": os.path.join(MODEL_DIR, "last_resnet.pth"),
    "hsemotion": os.path.join(MODEL_DIR, "last_hsemotion.pth"),
}

# ============================================================
# Dataset Configuration (Modular: for different datasets)
# ============================================================
DATASET_CONFIGS = {
    "fer2013": {
        "name": "FER2013",
        "data_dir": FER2013_DIR,
        "img_size": 48,
        "num_channels": 1,
        "num_classes": 7,
        "labels": EMOTION_LABELS
    },
    "ferplus": {
        "name": "FER+",
        "data_dir": FERPLUS_DIR,
        "img_size": 48,
        "num_channels": 1,
        "num_classes": 7,
        "labels": EMOTION_LABELS
    },
    "ckplus": {
        "name": "CK+",
        "data_dir": CKPLUS_DIR,
        "img_size": 48,
        "num_channels": 1,
        "num_classes": 7,
        "labels": EMOTION_LABELS
    },
    "rafdb": {
        "name": "RAF-DB",
        "data_dir": RAFDB_DIR,
        "img_size": 48,
        "num_channels": 1,
        "num_classes": 7,
        "labels": EMOTION_LABELS
    },
}
