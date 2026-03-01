"""
evaluate.py - Model Değerlendirme
==================================
Eğitilmiş modelin test seti üzerinde kapsamlı değerlendirmesi.
Confusion matrix, classification report, per-class accuracy.

Kullanım:
    python evaluate.py
    veya
    python main.py --mode evaluate
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

import config
from data.dataset import get_dataloaders, get_dataloaders_for_dataset
from models.mini_xception import get_model
from utils.visualization import plot_confusion_matrix


def evaluate_model(model=None, dataloader=None, split_name="Test",
                   dataset_name="fer2013"):
    """
    Modeli belirtilen veri seti üzerinde değerlendirir.

    Parametreler:
        model: PyTorch model (None ise en iyi model yüklenir)
        dataloader: Test DataLoader (None ise otomatik oluşturulur)
        split_name (str): Split adı (yazdırma için)

    Returns:
        dict: {
            'accuracy': float,
            'y_true': array,
            'y_pred': array,
            'report': str (classification report)
        }
    """
    # Model yükle
    if model is None:
        model = get_model(pretrained_path=config.BEST_MODEL_PATH)
    model.to(config.DEVICE)
    model.eval()

    # DataLoader
    if dataloader is None:
        dataloaders = get_dataloaders_for_dataset(dataset_name)
        dataloader = dataloaders['test']

    print(f"\n{'=' * 60}")
    print(f"  MODEL DEĞERLENDİRME ({split_name} Seti)")
    print(f"{'=' * 60}")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Değerlendirme", leave=True)

        for images, labels in pbar:
            images = images.to(config.DEVICE)

            # İleri yönlü geçiş
            outputs = model(images)

            # En yüksek olasılıklı sınıfı seç
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    # Genel doğruluk
    accuracy = accuracy_score(y_true, y_pred) * 100

    # Sınıf isimleri
    target_names = [config.EMOTION_LABELS[i] for i in range(config.NUM_CLASSES)]

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )

    # Sonuçları yazdır
    print(f"\n{'─' * 50}")
    print(f"  Genel Doğruluk: {accuracy:.2f}%")
    print(f"{'─' * 50}")
    print(f"\n  Classification Report:")
    print(report)

    # Confusion matrix çiz
    plot_confusion_matrix(y_true, y_pred)

    # Normalize edilmemiş versiyonu da kaydet
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(config.PLOT_DIR, "confusion_matrix_counts.png"),
        normalize=False
    )

    print(f"{'=' * 60}\n")

    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'report': report
    }


def compare_models(model_paths, model_names=None, dataset_name="fer2013"):
    """
    Birden fazla modeli karşılaştırır.
    İleride farklı datasetler veya mimariler karşılaştırılırken kullanılır.

    Parametreler:
        model_paths (list): Model dosya yolları
        model_names (list): Model isimleri (yazdırma için)
    """
    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(model_paths))]

    results = []

    dataloaders = get_dataloaders_for_dataset(dataset_name)

    for path, name in zip(model_paths, model_names):
        print(f"\n{'#' * 60}")
        print(f"  {name} değerlendiriliyor...")
        print(f"{'#' * 60}")

        model = get_model(pretrained_path=path)
        result = evaluate_model(model, dataloaders['test'], name)
        result['name'] = name
        results.append(result)

    # Karşılaştırma tablosu
    print(f"\n{'=' * 60}")
    print(f"  MODEL KARŞILAŞTIRMASI")
    print(f"{'=' * 60}")
    print(f"\n{'Model':<20} {'Accuracy':>10}")
    print(f"{'─' * 32}")
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:>9.2f}%")
    print(f"{'─' * 32}\n")

    return results


def main(dataset_name="fer2013"):
    """Doğrudan çalıştırma için."""
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"[HATA] Eğitilmiş model bulunamadı: {config.BEST_MODEL_PATH}")
        print("[HATA] Önce 'python train.py' ile modeli eğitin.")
        return

    evaluate_model(dataset_name=dataset_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Emotion Recognition Değerlendirme")
    parser.add_argument("--dataset", type=str, default="fer2013",
                        choices=["fer2013", "rafdb", "ckplus"],
                        help="Değerlendirme dataseti (varsayılan: fer2013)")
    args = parser.parse_args()
    main(dataset_name=args.dataset)
