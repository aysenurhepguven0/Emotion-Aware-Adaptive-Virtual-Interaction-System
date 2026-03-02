"""
main.py - Main Runner Script
=================================
Manages all functions from a single entry point.
Different modes can be selected via command line arguments.

Usage:
    python main.py --mode explore    # Dataset inceleme ve görselleştirme
    python main.py --mode train      # Model eğitimi
    python main.py --mode evaluate   # Test seti değerlendirmesi
    python main.py --mode predict --image yüz.jpg  # Tekil tahmin
    python main.py --mode all        # Run all sequentially
"""

import argparse
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def explore_dataset():
    """
    Loads, explores, and visualizes the dataset.

    Steps:
    1. Scan FER2013 folders
    2. Calculate and plot class distribution
    3. Show sample images from each class
    4. Print basic statistics
    """
    print("\n" + "=" * 60)
    print("  ADIM 1: DATASET İNCELEME")
    print("=" * 60)

    # Klasör varlık kontrolü
    if not os.path.exists(config.FER2013_DIR):
        print(f"\n[HATA] FER2013 klasörü bulunamadı!")
        print(f"  Beklenen konum: {config.FER2013_DIR}")
        print(f"\n  Çözüm:")
        print(f"  1. Kaggle'dan FER2013 datasetini indirin:")
        print(f"     https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"  2. İndirilen fer2013 klasörünü şu dizine koyun:")
        print(f"     {config.DATA_DIR}")
        return False

    from data.dataset import get_class_distribution, FER2013Dataset
    from utils.visualization import plot_class_distribution, plot_sample_images

    # 1. Sınıf dağılımını hesapla
    print("\n[1/3] Sınıf dağılımı hesaplanıyor...")
    distribution = get_class_distribution()

    # 2. Dağılım grafiğini çiz
    print("[2/3] Dağılım grafiği oluşturuluyor...")
    plot_class_distribution(distribution)

    # 3. Örnek görüntüler göster
    print("[3/3] Örnek görüntüler oluşturuluyor...")
    train_dir = os.path.join(config.FER2013_DIR, "train")
    train_dataset = FER2013Dataset(train_dir)
    plot_sample_images(train_dataset)

    print("\n[OK] Dataset exploration complete!")
    print(f"   Grafikler: {config.PLOT_DIR}")

    return True


def train_model(dataset_name="fer2013", model_name="mini_xception"):
    """
    Trains the model.

    Steps:
    1. Create DataLoaders
    2. Initialize selected model
    3. Run training loop
    4. Plot loss/accuracy graphs
    5. Save best model
    """
    print("\n" + "=" * 60)
    print(f"  STEP 2: MODEL TRAINING ({model_name})")
    print("=" * 60)

    # Dataset folder check
    if not os.path.exists(config.FER2013_DIR):
        print(f"\n[ERROR] FER2013 directory not found: {config.FER2013_DIR}")
        return False

    from train import Trainer

    trainer = Trainer(dataset_name=dataset_name, model_name=model_name)
    history = trainer.train()

    best_path = config.BEST_MODEL_PATHS.get(model_name, config.BEST_MODEL_PATH)
    print(f"\n[OK] Model training completed!")
    print(f"   Best model: {best_path}")
    print(f"   Plots: {config.PLOT_DIR}")

    return True


def evaluate_model(model_name="mini_xception"):
    """
    Evaluates trained model on test set.

    Steps:
    1. Load best model
    2. Run on test set
    3. Generate confusion matrix
    4. Print classification report
    """
    print("\n" + "=" * 60)
    print("  ADIM 3: MODEL DEĞERLENDİRME")
    print("=" * 60)

    # Model dosyası kontrolü - model-specific path kullan
    model_path = config.BEST_MODEL_PATHS.get(model_name, config.BEST_MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"\n[HATA] Eğitilmiş model bulunamadı: {model_path}")
        print("[HATA] First train the model with 'python main.py --mode train'.")
        return False

    from evaluate import evaluate_model as eval_fn
    from models.mini_xception import get_model

    model = get_model(pretrained_path=model_path)
    results = eval_fn(model=model)

    print("\n[OK] Evaluation complete!")
    print(f"   Doğruluk: {results['accuracy']:.2f}%")
    print(f"   Confusion matrix: {config.PLOT_DIR}")

    return True


def predict_emotion(image_path, model_name="mini_xception"):
    """
    Predicts emotion from a single image.

    Args:
        image_path (str): Face image file path
        model_name (str): Model architecture name
    """
    print("\n" + "=" * 60)
    print("  ADIM 4: DUYGU TAHMİNİ")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"\n[HATA] Görüntü bulunamadı: {image_path}")
        return False

    model_path = config.BEST_MODEL_PATHS.get(model_name, config.BEST_MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"\n[HATA] Eğitilmiş model bulunamadı: {model_path}")
        return False

    from inference import EmotionPredictor

    predictor = EmotionPredictor(model_path=model_path)
    result = predictor.predict_from_image(image_path)

    print(f"\n  Tahmin Sonucu:")
    print(f"  -----------------------------------")
    print(f"  Duygu:    {result['emotion']}")
    print(f"  Güven:    {result['confidence'] * 100:.1f}%")
    print(f"\n  Tüm Olasılıklar:")
    for emotion, prob in sorted(result['probabilities'].items(),
                                 key=lambda x: -x[1]):
        bar = '#' * int(prob * 30)
        print(f"    {emotion:<12} {prob * 100:5.1f}% {bar}")

    return True


def main():
    """Process command line arguments and run appropriate mode."""

    parser = argparse.ArgumentParser(
        description="FER2013 Duygu Tanıma Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --mode explore           Dataset inceleme
  python main.py --mode train             Model eğitimi
  python main.py --mode evaluate          Test değerlendirmesi
  python main.py --mode predict -i yüz.jpg  Tekil tahmin
  python main.py --mode webcam            Webcam ile canlı test
  python main.py --mode all               Run all sequentially
        """
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["explore", "train", "evaluate", "predict", "webcam", "all"],
        default="explore",
        help="Operation mode (default: explore)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["mini_xception", "efficientnet", "resnet", "hsemotion"],
        default="mini_xception",
        help="Model architecture (default: mini_xception)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fer2013", "ferplus", "rafdb", "ckplus"],
        default="fer2013",
        help="Training dataset (default: fer2013)"
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Image path for prediction (predict mode)"
    )

    args = parser.parse_args()

    # Header
    print("\n" + "=" * 60)
    print("  Emotion Recognition System")
    print(f"  Model: {args.model} | Dataset: {args.dataset}")
    print("=" * 60)

    print(f"\n  Project dir:  {config.PROJECT_ROOT}")
    print(f"  Device:       {config.DEVICE}")
    print(f"  Python:       {sys.version.split()[0]}")

    # Seçilen modu çalıştır
    if args.mode == "explore":
        explore_dataset()

    elif args.mode == "train":
        train_model(dataset_name=args.dataset, model_name=args.model)

    elif args.mode == "evaluate":
        evaluate_model(model_name=args.model)

    elif args.mode == "predict":
        if args.image is None:
            print("\n[HATA] predict modu için --image argümanı gerekli!")
            print("  Örnek: python main.py --mode predict --image yüz.jpg")
        else:
            predict_emotion(args.image, model_name=args.model)

    elif args.mode == "webcam":
        model_path = config.BEST_MODEL_PATHS.get(args.model, config.BEST_MODEL_PATH)
        if not os.path.exists(model_path):
            print(f"\n[HATA] Eğitilmiş model bulunamadı: {model_path}")
            print("[HATA] First train the model with 'python main.py --mode train'.")
        else:
            from webcam import WebcamEmotionDetector
            detector = WebcamEmotionDetector(model_path=model_path)
            detector.run()

    elif args.mode == "all":
        print("\n  Tüm adımlar sırayla çalıştırılacak...")
        print("  ---------------------------------------")

        # Adım 1: Dataset inceleme
        success = explore_dataset()
        if not success:
            return

        # Adım 2: Model eğitimi
        success = train_model()
        if not success:
            return

        # Adım 3: Değerlendirme
        success = evaluate_model()
        if not success:
            return

        print("\n" + "=" * 60)
        print("  ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 60)


if __name__ == "__main__":
    main()
