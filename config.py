"""
config.py - Proje Konfigürasyonu
================================
Tüm hyperparameter'lar, yollar ve sabitler burada tanımlanır.
Farklı datasetler için kolayca genişletilebilir yapıdadır.
"""

import os
import torch

# ============================================================
# Proje Dizin Yolları
# ============================================================
# Bu dosyanın bulunduğu dizin = proje kökü
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Veri yolları
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FER2013_DIR = os.path.join(DATA_DIR, "fer2013")  # Klasör tabanlı dataset
RAFDB_DIR = os.path.join(DATA_DIR, "raf-db")     # RAF-DB dataset
CKPLUS_DIR = os.path.join(DATA_DIR, "ck+")       # CK+ dataset

# Çıktı dizinleri
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Çıktı dizinlerini oluştur
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# Cihaz Ayarları
# ============================================================
# GPU varsa GPU kullan, yoksa CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# FER2013 Dataset Bilgileri
# ============================================================
# 6 Ekman temel duygu sınıfı (Neutral hariç)
EMOTION_LABELS = {
    0: "Angry",      # Kızgın
    1: "Disgust",    # Tiksinme
    2: "Fear",       # Korku
    3: "Happy",      # Mutlu
    4: "Sad",        # Üzgün
    5: "Surprise",   # Şaşkın
}

NUM_CLASSES = len(EMOTION_LABELS)  # 6
IMG_SIZE = 48  # FER2013 görüntü boyutu: 48x48 piksel
NUM_CHANNELS = 1  # Gri tonlama

# RAF-DB etiketleri (orijinal 1-7 → standart 0-5 mapping, Neutral hariç)
RAFDB_LABEL_MAP = {
    1: 5,  # Surprise  → 5
    2: 2,  # Fear      → 2
    3: 1,  # Disgust   → 1
    4: 3,  # Happiness → 3
    5: 4,  # Sadness   → 4
    6: 0,  # Anger     → 0
    # 7: Neutral → filtre edilecek
}

# CK+ etiketleri (8 sınıf — Contempt ek sınıf)
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

# CK+ → 6 Ekman mapping (Contempt ve Neutral çıkarılır)
CKPLUS_TO_FER_MAP = {
    0: 0,  # Anger    → Angry
    2: 1,  # Disgust  → Disgust
    7: 2,  # Fear     → Fear
    3: 3,  # Happy    → Happy
    4: 4,  # Sadness  → Sad
    5: 5,  # Surprise → Surprise
    # 1: Contempt → çıkarılır
    # 6: Neutral  → çıkarılır
}

# ============================================================
# Eğitim Hyperparameter'ları
# ============================================================
BATCH_SIZE = 64       # i5 + 4GB RAM için uygun
EPOCHS = 50           # Maksimum epoch sayısı
LEARNING_RATE = 1e-3  # Adam optimizer için başlangıç öğrenme oranı
WEIGHT_DECAY = 1e-4   # L2 regularization

# Learning Rate Scheduler
LR_PATIENCE = 5       # Kaç epoch iyileşme olmazsa LR düşür
LR_FACTOR = 0.5       # LR'yi bu faktörle çarp

# Early Stopping
EARLY_STOPPING_PATIENCE = 10  # Kaç epoch iyileşme olmazsa dur

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
# Model Seçimi
# ============================================================
# Kullanılabilir modeller: "mini_xception", "efficientnet"
MODEL_NAME = "mini_xception"  # Varsayılan model

# EfficientNet Transfer Learning Ayarları
EFFICIENTNET_FREEZE_BACKBONE = True    # Backbone'u dondur
EFFICIENTNET_UNFREEZE_LAST_N = 2       # Son N bloğu çöz
EFFICIENTNET_LR = 1e-4                 # Transfer Learning için düşük LR

# ============================================================
# Model Kaydetme
# ============================================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pth")

# Model-spesifik yollar
BEST_MODEL_PATHS = {
    "mini_xception": os.path.join(MODEL_DIR, "best_mini_xception.pth"),
    "efficientnet": os.path.join(MODEL_DIR, "best_efficientnet.pth"),
}
LAST_MODEL_PATHS = {
    "mini_xception": os.path.join(MODEL_DIR, "last_mini_xception.pth"),
    "efficientnet": os.path.join(MODEL_DIR, "last_efficientnet.pth"),
}

# ============================================================
# Dataset Yapılandırması (Modüler: farklı datasetler için)
# ============================================================
DATASET_CONFIGS = {
    "fer2013": {
        "name": "FER2013",
        "data_dir": FER2013_DIR,
        "img_size": 48,
        "num_channels": 1,
        "num_classes": 6,
        "labels": EMOTION_LABELS
    },
    "ckplus": {
        "name": "CK+",
        "data_dir": CKPLUS_DIR,
        "img_size": 48,
        "num_channels": 1,
        "num_classes": 6,
        "labels": EMOTION_LABELS
    },
    "rafdb": {
        "name": "RAF-DB",
        "data_dir": RAFDB_DIR,
        "img_size": 48,
        "num_channels": 1,
        "num_classes": 6,
        "labels": EMOTION_LABELS
    },
}
