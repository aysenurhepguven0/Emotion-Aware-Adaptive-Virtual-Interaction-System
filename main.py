"""
main.py - Ana Çalıştırma Scripti
=================================
Tüm işlevleri tek bir yerden yönetir.
Komut satırı argümanları ile farklı modlar seçilebilir.

Kullanım:
    python main.py --mode explore    # Dataset inceleme ve görselleştirme
    python main.py --mode train      # Model eğitimi
    python main.py --mode evaluate   # Test seti değerlendirmesi
    python main.py --mode predict --image yüz.jpg  # Tekil tahmin
    python main.py --mode all        # Hepsini sırayla çalıştır
"""

import argparse
import os
import sys

# Proje kök dizinini Python path'ine ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def explore_dataset():
    """
    Dataset'i yükler, inceler ve görselleştirir.

    Adımlar:
    1. FER2013 klasörlerini tara
    2. Sınıf dağılımını hesapla ve çiz
    3. Her sınıftan örnek görüntüler göster
    4. Temel istatistikleri yazdır
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

    print("\n✅ Dataset inceleme tamamlandı!")
    print(f"   Grafikler: {config.PLOT_DIR}")

    return True


def train_model():
    """
    Modeli eğitir.

    Adımlar:
    1. DataLoader'ları oluştur
    2. Mini-Xception modelini başlat
    3. Eğitim döngüsünü çalıştır
    4. Loss/accuracy grafiklerini çiz
    5. En iyi modeli kaydet
    """
    print("\n" + "=" * 60)
    print("  ADIM 2: MODEL EĞİTİMİ")
    print("=" * 60)

    # Dataset klasör kontrolü
    if not os.path.exists(config.FER2013_DIR):
        print(f"\n[HATA] FER2013 klasörü bulunamadı: {config.FER2013_DIR}")
        return False

    from train import Trainer

    trainer = Trainer()
    history = trainer.train()

    print("\n✅ Model eğitimi tamamlandı!")
    print(f"   En iyi model: {config.BEST_MODEL_PATH}")
    print(f"   Grafikler: {config.PLOT_DIR}")

    return True


def evaluate_model():
    """
    Eğitilmiş modeli test seti üzerinde değerlendirir.

    Adımlar:
    1. En iyi modeli yükle
    2. Test setinde çalıştır
    3. Confusion matrix oluştur
    4. Classification report yazdır
    """
    print("\n" + "=" * 60)
    print("  ADIM 3: MODEL DEĞERLENDİRME")
    print("=" * 60)

    # Model dosyası kontrolü
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"\n[HATA] Eğitilmiş model bulunamadı: {config.BEST_MODEL_PATH}")
        print("[HATA] Önce 'python main.py --mode train' ile modeli eğitin.")
        return False

    from evaluate import evaluate_model as eval_fn

    results = eval_fn()

    print("\n✅ Değerlendirme tamamlandı!")
    print(f"   Doğruluk: {results['accuracy']:.2f}%")
    print(f"   Confusion matrix: {config.PLOT_DIR}")

    return True


def predict_emotion(image_path):
    """
    Tekil görüntüden duygu tahmini yapar.

    Parametreler:
        image_path (str): Yüz görüntüsü dosya yolu
    """
    print("\n" + "=" * 60)
    print("  ADIM 4: DUYGU TAHMİNİ")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"\n[HATA] Görüntü bulunamadı: {image_path}")
        return False

    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"\n[HATA] Eğitilmiş model bulunamadı: {config.BEST_MODEL_PATH}")
        return False

    from inference import EmotionPredictor

    predictor = EmotionPredictor()
    result = predictor.predict_from_image(image_path)

    print(f"\n  Tahmin Sonucu:")
    print(f"  ─────────────────────────────────")
    print(f"  Duygu:    {result['emotion']}")
    print(f"  Güven:    {result['confidence'] * 100:.1f}%")
    print(f"\n  Tüm Olasılıklar:")
    for emotion, prob in sorted(result['probabilities'].items(),
                                 key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {emotion:<12} {prob * 100:5.1f}% {bar}")

    return True


def main():
    """Komut satırı argümanlarını işle ve uygun modu çalıştır."""

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
  python main.py --mode all               Hepsini sırayla çalıştır
        """
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["explore", "train", "evaluate", "predict", "webcam", "all"],
        default="explore",
        help="Çalışma modu (varsayılan: explore)"
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Tahmin yapılacak görüntü yolu (predict modu için)"
    )

    args = parser.parse_args()

    # Başlık
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  FER2013 Yüz İfadelerinden Duygu Tanıma Sistemi".center(58) + "║")
    print("║" + "  Bitirme Projesi - Mini-Xception (PyTorch)".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    print(f"\n  Proje dizini: {config.PROJECT_ROOT}")
    print(f"  Cihaz:        {config.DEVICE}")
    print(f"  Python:       {sys.version.split()[0]}")

    # Seçilen modu çalıştır
    if args.mode == "explore":
        explore_dataset()

    elif args.mode == "train":
        train_model()

    elif args.mode == "evaluate":
        evaluate_model()

    elif args.mode == "predict":
        if args.image is None:
            print("\n[HATA] predict modu için --image argümanı gerekli!")
            print("  Örnek: python main.py --mode predict --image yüz.jpg")
        else:
            predict_emotion(args.image)

    elif args.mode == "webcam":
        if not os.path.exists(config.BEST_MODEL_PATH):
            print(f"\n[HATA] Eğitilmiş model bulunamadı: {config.BEST_MODEL_PATH}")
            print("[HATA] Önce 'python main.py --mode train' ile modeli eğitin.")
        else:
            from webcam import WebcamEmotionDetector
            detector = WebcamEmotionDetector()
            detector.run()

    elif args.mode == "all":
        print("\n  Tüm adımlar sırayla çalıştırılacak...")
        print("  ─────────────────────────────────────")

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

        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + "  ✅ TÜM ADIMLAR BAŞARIYLA TAMAMLANDI!".center(58) + "║")
        print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
