"""
data/dataset.py - FER2013 Dataset Yükleme ve Hazırlama
======================================================
FER2013 görüntü klasörlerinden veri okuma, PyTorch Dataset sınıfı,
DataLoader oluşturma ve veri dağılımı analizi.

FER2013 Klasör Yapısı (Kaggle image format):
    fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── ...

Her alt klasörde 48x48 gri tonlamalı JPG görüntüler bulunur.
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


# Klasör adlarından duygu etiketlerine mapping (6 Ekman, Neutral hariç)
FOLDER_TO_LABEL = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    # "neutral" çıkarıldı — 6 Ekman duygusuna odaklanıyoruz
}


class FER2013Dataset(Dataset):
    """
    FER2013 veri seti için PyTorch Dataset sınıfı (klasör tabanlı).

    Parametreler:
        root_dir (str): Görüntü kök dizini (örn: data/fer2013/train)
        transform (callable, optional): Görüntüye uygulanacak dönüşümler
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Tüm görüntü yollarını ve etiketlerini topla
        self.image_paths = []
        self.labels = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"Veri dizini bulunamadı: {root_dir}\n"
                f"FER2013 datasetini data/fer2013/ altına yerleştirin."
            )

        # Her duygu klasörünü tara
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Klasör adını etikete çevir
            label_name = folder_name.lower()
            if label_name not in FOLDER_TO_LABEL:
                print(f"[UYARI] Bilinmeyen klasör atlandı: {folder_name}")
                continue

            label = FOLDER_TO_LABEL[label_name]

            # Klasördeki tüm görüntüleri topla
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for img_file in image_files:
                self.image_paths.append(os.path.join(folder_path, img_file))
                self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"[INFO] {root_dir}: {len(self.image_paths)} görüntü yüklendi "
              f"({len(set(self.labels))} sınıf)")

    def __len__(self):
        """Dataset'teki toplam örnek sayısı."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Belirtilen index'teki görüntü ve etiketi döndürür.

        Returns:
            image (Tensor): [1, 48, 48] boyutunda normalize edilmiş görüntü
            label (int): 0-5 arası duygu etiketi (6 Ekman)
        """
        # Görüntüyü oku
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Gri tonlamaya çevir (zaten gri olan JPG'ler için de güvenli)
        if image.mode != 'L':
            image = image.convert('L')

        # 48x48'e boyutlandır (normalde zaten 48x48)
        if image.size != (config.IMG_SIZE, config.IMG_SIZE):
            image = image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)

        # Numpy array'e dönüştür ve normalize et [0, 255] → [0.0, 1.0]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # (48, 48) → (1, 48, 48) kanal boyutu ekle
        img_array = np.expand_dims(img_array, axis=0)

        # Tensor'a dönüştür
        tensor = torch.FloatTensor(img_array)

        # Augmentation uygula (varsa)
        if self.transform:
            tensor = self.transform(tensor)

        label = torch.LongTensor([self.labels[idx]]).squeeze()

        return tensor, label

    def get_sample_images(self, num_per_class=3):
        """
        Her sınıftan örnek görüntüler döndürür (NumPy array olarak).

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


def get_transforms(split="train"):
    """
    Split'e göre uygun veri dönüşümlerini döndürür.

    - train: Data augmentation (flip, rotation, affine, erasing)
    - test/val: Augmentation yok

    Parametreler:
        split (str): 'train' veya 'test'

    Returns:
        transforms.Compose veya None
    """
    if split == "train":
        return transforms.Compose([
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
        ])
    else:
        return None


def get_dataloaders(data_dir=None, batch_size=None, val_split=0.15):
    """
    Train, Validation ve Test DataLoader'larını oluşturur.

    Klasör yapısındaki FER2013:
    - train/  →  Eğitim + Doğrulama (%85 / %15 random split)
    - test/   →  Test seti

    Parametreler:
        data_dir (str): FER2013 kök dizini (varsayılan: config'den)
        batch_size (int): Batch boyutu
        val_split (float): Eğitim setinden doğrulamaya ayrılacak oran

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    if data_dir is None:
        data_dir = config.FER2013_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    print("\n" + "=" * 60)
    print("  FER2013 Dataset Yükleniyor (Klasör Formatı)")
    print("=" * 60)

    # Train seti — augmentation ile
    full_train_dataset = FER2013Dataset(train_dir, transform=None)

    # Train setini Train + Validation olarak böl
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Tekrarlanabilirlik
    )

    # Augmentation: sadece train subset'e uygulanır
    # (random_split sonrası transform eklemek için wrapper kullanıyoruz)
    train_dataset = TransformSubset(train_subset, get_transforms("train"))
    val_dataset = TransformSubset(val_subset, None)

    # Test seti — augmentation yok
    test_dataset = FER2013Dataset(test_dir, transform=None)

    # DataLoader'lar
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,       # Windows uyumluluğu
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

    # Özet bilgi
    print(f"\n[INFO] Dataset boyutları:")
    print(f"  Train:      {len(train_dataset):,} örnek")
    print(f"  Validation: {len(val_dataset):,} örnek")
    print(f"  Test:       {len(test_dataset):,} örnek")
    print(f"  Batch:      {batch_size}")
    print(f"  Cihaz:      {config.DEVICE}")
    print("=" * 60 + "\n")

    return dataloaders


class TransformSubset:
    """
    random_split sonrası subset'e transform uygulamak için wrapper.
    PyTorch'un Subset'i doğrudan transform desteklemediği için gerekli.
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
    Her split için sınıf dağılımını hesaplar.

    Returns:
        dict: {'train': {0: count, ...}, 'test': {0: count, ...}}
    """
    if data_dir is None:
        data_dir = config.FER2013_DIR

    distribution = {}

    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"[UYARI] {split_dir} bulunamadı, atlanıyor.")
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

    # Dağılımı yazdır
    print("\n" + "=" * 60)
    print("  Sınıf Dağılımı (Class Distribution)")
    print("=" * 60)

    splits = list(distribution.keys())
    header = f"{'Sınıf':<12}" + "".join(f"{s:>10}" for s in splits)
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

    # Toplam
    row = f"{'TOPLAM':<12}"
    for split in splits:
        total = sum(distribution[split].values())
        row += f"{total:>10,}"
    print(row)
    print("=" * 60 + "\n")

    return distribution


def get_class_weights(data_dir=None):
    """
    Dengesiz sınıf dağılımını dengelemek için sınıf ağırlıkları hesaplar.
    Az örnekli sınıflar (örn. Disgust) daha yüksek ağırlık alır.

    Returns:
        torch.Tensor: Her sınıf için ağırlık tensörü
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

    # Ters frekans ağırlıklandırma
    weights = []
    for i in range(config.NUM_CLASSES):
        if i in counts and counts[i] > 0:
            w = total / (num_classes_present * counts[i])
        else:
            w = 1.0  # Sınıf yoksa varsayılan ağırlık
        weights.append(w)

    weights_tensor = torch.FloatTensor(weights)

    print("[INFO] Sınıf ağırlıkları (class weights):")
    for i, w in enumerate(weights):
        count = counts.get(i, 0)
        print(f"  {config.EMOTION_LABELS[i]:<12}: {w:.4f}  ({count:,} örnek)")

    return weights_tensor


# ================================================================
# RAF-DB Dataset
# ================================================================
class RAFDBDataset(Dataset):
    """
    RAF-DB veri seti için PyTorch Dataset sınıfı.

    RAF-DB yapısı:
        raf-db/
        ├── DATASET/
        │   ├── train/
        │   │   ├── 1/   (Surprise)
        │   │   ├── 2/   (Fear)
        │   │   ├── ...
        │   │   └── 7/   (Neutral)
        │   └── test/
        │       ├── 1/ ... 7/
        ├── train_labels.csv
        └── test_labels.csv

    Görüntüler 100x100 RGB → 48x48 grayscale'e dönüştürülür.
    Label mapping: RAF-DB (1-7) → FER2013 standart (0-6)
    """

    def __init__(self, root_dir, split="train", transform=None):
        """
        Parametreler:
            root_dir (str): RAF-DB kök dizini (data/raf-db)
            split (str): 'train' veya 'test'
            transform (callable): Görüntü dönüşümleri
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        img_dir = os.path.join(root_dir, "DATASET", split)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(
                f"RAF-DB dizini bulunamadı: {img_dir}\n"
                f"RAF-DB datasetini data/raf-db/ altına yerleştirin."
            )

        # Klasör bazlı okuma (1-7 numaralı klasörler)
        for class_folder in sorted(os.listdir(img_dir)):
            class_path = os.path.join(img_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            try:
                raf_label = int(class_folder)
            except ValueError:
                continue

            if raf_label not in config.RAFDB_LABEL_MAP:
                print(f"[UYARI] Bilinmeyen RAF-DB etiketi: {raf_label}")
                continue

            fer_label = config.RAFDB_LABEL_MAP[raf_label]

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(fer_label)

        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"[INFO] RAF-DB {split}: {len(self.image_paths)} görüntü yüklendi "
              f"({len(set(self.labels))} sınıf)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """100x100 RGB → 48x48 grayscale tensor döndürür."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # RGB → Grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # 48x48'e resize
        if image.size != (config.IMG_SIZE, config.IMG_SIZE):
            image = image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)

        # Normalize et
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 48, 48)

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
    CK+ veri seti için PyTorch Dataset sınıfı.

    CSV formatı (FER2013 benzeri):
        emotion,pixels,Usage
        6, 36 39 35 ..., Training

    - Contempt sınıfı (label=1) ve Neutral sınıfı (label=6) çıkarılır.
    - CK+ etiketleri 6 Ekman etiketlerine dönüştürülür.
    - Piksel verisi 48x48 grayscale.
    """

    def __init__(self, csv_path, split="train", transform=None):
        """
        Parametreler:
            csv_path (str): CK+ CSV dosya yolu
            split (str): 'train' veya 'test' (Usage sütunundan filtrelenir)
            transform (callable): Görüntü dönüşümleri
        """
        import pandas as pd

        self.transform = transform
        self.images = []
        self.labels = []

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CK+ CSV bulunamadı: {csv_path}\n"
                f"ckextended.csv dosyasını data/ck+/ altına yerleştirin."
            )

        df = pd.read_csv(csv_path)

        # Usage sütununa göre filtrele
        usage_map = {"train": "Training", "test": "Testing",
                      "val": "PublicTest"}
        usage_filter = usage_map.get(split, "Training")

        if "Usage" in df.columns:
            df = df[df["Usage"] == usage_filter]

        skipped_contempt = 0

        for _, row in df.iterrows():
            ck_emotion = int(row["emotion"])

            # Contempt sınıfını atla
            if ck_emotion not in config.CKPLUS_TO_FER_MAP:
                skipped_contempt += 1
                continue

            fer_label = config.CKPLUS_TO_FER_MAP[ck_emotion]

            # Piksel verisini 48x48 numpy array'e çevir
            pixels = np.array(
                row["pixels"].split(), dtype=np.float32
            ).reshape(config.IMG_SIZE, config.IMG_SIZE)

            self.images.append(pixels)
            self.labels.append(fer_label)

        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        info = f"[INFO] CK+ {split}: {len(self.images)} görüntü yüklendi "
        info += f"({len(set(self.labels))} sınıf)"
        if skipped_contempt > 0:
            info += f" [{skipped_contempt} Contempt/Neutral örneği atlandı]"
        print(info)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """48x48 grayscale tensor döndürür."""
        image = self.images[idx] / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # (1, 48, 48)

        tensor = torch.FloatTensor(image)

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return tensor, label


# ================================================================
# Fabrika Fonksiyonu — Dataset'e göre DataLoader oluştur
# ================================================================
def get_dataloaders_for_dataset(dataset_name, batch_size=None, val_split=0.15):
    """
    Belirtilen dataset için Train/Val/Test DataLoader'ları oluşturur.

    Parametreler:
        dataset_name (str): 'fer2013', 'rafdb', veya 'ckplus'
        batch_size (int): Batch boyutu
        val_split (float): Eğitim setinden validation'a ayrılacak oran

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset_name = dataset_name.lower()

    if dataset_name == "fer2013":
        return get_dataloaders(batch_size=batch_size, val_split=val_split)

    elif dataset_name == "rafdb":
        return _get_rafdb_dataloaders(batch_size, val_split)

    elif dataset_name == "ckplus":
        return _get_ckplus_dataloaders(batch_size, val_split)

    else:
        available = ", ".join(config.DATASET_CONFIGS.keys())
        raise ValueError(
            f"Bilinmeyen dataset: '{dataset_name}'. "
            f"Mevcut datasetler: {available}"
        )


def _get_rafdb_dataloaders(batch_size, val_split):
    """RAF-DB DataLoader'larını oluşturur."""
    print("\n" + "=" * 60)
    print("  RAF-DB Dataset Yükleniyor")
    print("=" * 60)

    # Train seti
    full_train = RAFDBDataset(config.RAFDB_DIR, split="train")

    # Train → Train + Validation split
    total = len(full_train)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_subset, val_subset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = TransformSubset(train_subset, get_transforms("train"))
    val_dataset = TransformSubset(val_subset, None)

    # Test seti
    test_dataset = RAFDBDataset(config.RAFDB_DIR, split="test")

    dataloaders = _build_dataloaders(train_dataset, val_dataset,
                                      test_dataset, batch_size)

    _print_summary("RAF-DB", train_dataset, val_dataset,
                    test_dataset, batch_size)
    return dataloaders


def _get_ckplus_dataloaders(batch_size, val_split):
    """CK+ DataLoader'larını oluşturur."""
    print("\n" + "=" * 60)
    print("  CK+ Dataset Yükleniyor")
    print("=" * 60)

    csv_path = os.path.join(config.CKPLUS_DIR, "ckextended.csv")

    # CK+ genellikle sadece Training içerir, biz split yapacağız
    full_dataset = CKPlusDataset(csv_path, split="train")

    total = len(full_dataset)
    test_size = int(total * 0.15)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = TransformSubset(train_subset, get_transforms("train"))
    val_dataset = TransformSubset(val_subset, None)
    test_dataset = TransformSubset(test_subset, None)

    dataloaders = _build_dataloaders(train_dataset, val_dataset,
                                      test_dataset, batch_size)

    _print_summary("CK+", train_dataset, val_dataset,
                    test_dataset, batch_size)
    return dataloaders


def _build_dataloaders(train_ds, val_ds, test_ds, batch_size):
    """Ortak DataLoader oluşturma yardımcısı."""
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
    """Ortak dataset özeti yazdırma."""
    print(f"\n[INFO] {name} Dataset boyutları:")
    print(f"  Train:      {len(train_ds):,} örnek")
    print(f"  Validation: {len(val_ds):,} örnek")
    print(f"  Test:       {len(test_ds):,} örnek")
    print(f"  Batch:      {batch_size}")
    print(f"  Cihaz:      {config.DEVICE}")
    print("=" * 60 + "\n")


def get_class_weights_for_dataset(dataset_name, data_dir=None):
    """
    Herhangi bir dataset için sınıf ağırlıklarını hesaplar.

    Parametreler:
        dataset_name (str): 'fer2013', 'rafdb', 'ckplus'

    Returns:
        torch.Tensor: Sınıf ağırlıkları
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

    else:
        raise ValueError(f"Bilinmeyen dataset: {dataset_name}")


def _compute_weights(counts):
    """Sınıf ağırlıklarını hesapla ve yazdır."""
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

    print("[INFO] Sınıf ağırlıkları (class weights):")
    for i, w in enumerate(weights):
        count = counts.get(i, 0)
        print(f"  {config.EMOTION_LABELS[i]:<12}: {w:.4f}  ({count:,} örnek)")

    return weights_tensor

