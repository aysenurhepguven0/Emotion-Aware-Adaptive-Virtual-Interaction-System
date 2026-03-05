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
# Emotion Labels (5 Emotions: Disgust & Fear removed)
# ============================================================
# 4 Ekman emotions (Angry, Happy, Sad, Surprise) + Neutral
EMOTION_LABELS = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Neutral",
}

NUM_CLASSES = len(EMOTION_LABELS)  # 5
IMG_SIZE = 48   # FER2013 image size: 48x48 pixels
NUM_CHANNELS = 1  # Grayscale

# RAF-DB labels (original 1-7 mapped to standard 0-4)
RAFDB_LABEL_MAP = {
    1: 3,  # Surprise  -> 3
    # 2: Fear      -> removed
    # 3: Disgust   -> removed
    4: 1,  # Happiness -> 1
    5: 2,  # Sadness   -> 2
    6: 0,  # Anger     -> 0
    7: 4,  # Neutral   -> 4
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

# CK+ -> 5 class mapping (Contempt, Disgust, Fear removed)
CKPLUS_TO_FER_MAP = {
    0: 0,  # Anger    -> Angry
    # 2: Disgust  -> removed
    # 7: Fear     -> removed
    3: 1,  # Happy    -> Happy
    4: 2,  # Sadness  -> Sad
    5: 3,  # Surprise -> Surprise
    6: 4,  # Neutral  -> Neutral
    # 1: Contempt -> removed
}

# FERPlus folder name -> 5 emotion label mapping
# Note: 'suprise' is the actual folder name in the dataset (misspelled)
FERPLUS_FOLDER_TO_LABEL = {
    "angry": 0,
    # "disgust" -> removed
    # "fear" -> removed
    "happy": 1,
    "sad": 2,
    "suprise": 3,    # Misspelled in dataset
    "surprise": 3,   # Fallback if corrected
    "neutral": 4,
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
        "num_classes": 5,
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
