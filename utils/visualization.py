"""
utils/visualization.py - Görselleştirme Fonksiyonları
=====================================================
Veri dağılımı, örnek görüntüler, eğitim grafikleri ve
confusion matrix görselleştirmeleri.

Tüm grafikler hem ekranda gösterilir hem de outputs/plots/ altına kaydedilir.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI gerektirmeyen backend (sunucu ortamı uyumlu)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Grafik stili ayarları
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_class_distribution(distribution, save_path=None):
    """
    Her split için sınıf dağılımını bar chart olarak çizer.

    Parametreler:
        distribution (dict): get_class_distribution() çıktısı
        save_path (str): Grafik kayıt yolu (varsayılan: outputs/plots/)
    """
    if save_path is None:
        save_path = os.path.join(config.PLOT_DIR, "class_distribution.png")

    splits = list(distribution.keys())  # ['train', 'test']
    num_splits = len(splits)
    split_labels = {"train": "Eğitim (Train)", "test": "Test"}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFD93D', '#C9B1FF']

    fig, axes = plt.subplots(1, num_splits, figsize=(9 * num_splits, 5))
    if num_splits == 1:
        axes = [axes]

    for idx, split in enumerate(splits):
        counts = [distribution[split].get(i, 0) for i in range(config.NUM_CLASSES)]
        labels = [config.EMOTION_LABELS[i] for i in range(config.NUM_CLASSES)]
        turkish_name = split_labels.get(split, split)

        bars = axes[idx].bar(labels, counts, color=colors, edgecolor='white', linewidth=0.5)
        axes[idx].set_title(f"{turkish_name}", fontsize=14, fontweight='bold')
        axes[idx].set_xlabel("Duygu Sınıfı")
        axes[idx].set_ylabel("Örnek Sayısı")
        axes[idx].tick_params(axis='x', rotation=45)

        # Bar üzerine sayı yaz
        for bar, count in zip(bars, counts):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=9
            )

    plt.suptitle("FER2013 Sınıf Dağılımı", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Sınıf dağılımı grafiği kaydedildi: {save_path}")


def plot_sample_images(dataset, num_per_class=3, save_path=None):
    """
    Her duygu sınıfından örnek görüntüler gösterir.

    Parametreler:
        dataset: FER2013Dataset instance
        num_per_class (int): Her sınıftan gösterilecek örnek sayısı
        save_path (str): Grafik kayıt yolu
    """
    from PIL import Image

    if save_path is None:
        save_path = os.path.join(config.PLOT_DIR, "sample_images.png")

    # Mevcut sınıfları bul
    present_classes = sorted(set(dataset.labels.tolist()))
    num_rows = len(present_classes)

    fig, axes = plt.subplots(
        num_rows, num_per_class,
        figsize=(num_per_class * 2.5, num_rows * 2.5)
    )
    if num_rows == 1:
        axes = [axes]  # Tek satır durumunda liste yap

    # Her sınıf için örnek bul
    for row, cls_id in enumerate(present_classes):
        # Bu sınıfa ait indeksleri bul
        class_indices = np.where(dataset.labels == cls_id)[0]

        # Rastgele seç
        if len(class_indices) >= num_per_class:
            selected = np.random.choice(class_indices, num_per_class, replace=False)
        else:
            selected = class_indices[:num_per_class]

        for col, idx in enumerate(selected):
            ax = axes[row][col] if num_rows > 1 else axes[col]

            # Görüntüyü dosyadan oku
            img = Image.open(dataset.image_paths[idx]).convert('L')
            ax.imshow(np.array(img), cmap='gray')
            ax.axis('off')

            # İlk sütuna sınıf adını yaz
            if col == 0:
                ax.set_ylabel(
                    config.EMOTION_LABELS[cls_id],
                    fontsize=12, fontweight='bold', rotation=0,
                    labelpad=60, ha='right', va='center'
                )

    plt.suptitle("FER2013 Örnek Görüntüler", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Örnek görüntüler kaydedildi: {save_path}")


def plot_training_history(history, save_path=None):
    """
    Eğitim sürecindeki loss ve accuracy değerlerini çizer.

    Parametreler:
        history (dict): {
            'train_loss': [...], 'val_loss': [...],
            'train_acc': [...], 'val_acc': [...],
            'lr': [...]
        }
        save_path (str): Grafik kayıt yolu
    """
    if save_path is None:
        save_path = os.path.join(config.PLOT_DIR, "training_history.png")

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ---- Loss Grafiği ----
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Eğitim', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Doğrulama', markersize=3)
    axes[0].set_title('Loss (Kayıp)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Accuracy Grafiği ----
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Eğitim', markersize=3)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Doğrulama', markersize=3)
    axes[1].set_title('Accuracy (Doğruluk)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ---- Learning Rate Grafiği ----
    if 'lr' in history and history['lr']:
        axes[2].plot(epochs, history['lr'], 'g-o', markersize=3)
        axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'LR verisi yok', ha='center', va='center',
                     transform=axes[2].transAxes, fontsize=14)

    plt.suptitle("Eğitim Süreci", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Eğitim grafikleri kaydedildi: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=True):
    """
    Confusion matrix'i heatmap olarak çizer.

    Parametreler:
        y_true (array): Gerçek etiketler
        y_pred (array): Tahmin edilen etiketler
        save_path (str): Grafik kayıt yolu
        normalize (bool): True ise yüzde olarak göster
    """
    if save_path is None:
        save_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")

    labels = [config.EMOTION_LABELS[i] for i in range(config.NUM_CLASSES)]

    # Confusion matrix hesapla
    cm = sk_confusion_matrix(y_true, y_pred)

    if normalize:
        # Her satırı toplama böl (yüzde)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.1%'
        title = "Confusion Matrix (Normalize Edilmiş)"
    else:
        cm_display = cm
        fmt = 'd'
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_display, annot=True, fmt=fmt,
        xticklabels=labels, yticklabels=labels,
        cmap='Blues', ax=ax,
        linewidths=0.5, linecolor='white',
        square=True,
        cbar_kws={'label': 'Oran' if normalize else 'Sayı'}
    )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Tahmin Edilen', fontsize=13)
    ax.set_ylabel('Gerçek', fontsize=13)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Confusion matrix kaydedildi: {save_path}")

    # Her sınıf için doğruluk oranını yazdır
    print("\n  Per-class Accuracy:")
    for i in range(config.NUM_CLASSES):
        if cm.sum(axis=1)[i] > 0:
            class_acc = cm[i, i] / cm.sum(axis=1)[i] * 100
            print(f"    {labels[i]:<12}: {class_acc:.1f}%")
