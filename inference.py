"""
inference.py - Tahmin Modülü (Web Entegrasyonu)
================================================
Eğitilmiş modeli kullanarak tekil görüntülerden duygu tahmini yapar.
Flask/FastAPI backend'e import edilerek doğrudan kullanılabilir.

Kullanım:
    # Python'dan:
    from inference import EmotionPredictor
    predictor = EmotionPredictor()
    result = predictor.predict_from_image("yüz.jpg")

    # Komut satırından:
    python inference.py --image yüz.jpg
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from models.mini_xception import get_model


class EmotionPredictor:
    """
    Duygu tahmin sınıfı.
    Eğitilmiş modeli yükler ve yeni görüntüler üzerinde tahmin yapar.

    Bu sınıf web backend'e entegre edilmek üzere tasarlanmıştır:
    - Bir kez model yüklenir (constructor'da)
    - Her istek için predict metodu çağrılır
    - Thread-safe değildir (her worker için ayrı instance önerilir)

    Örnek Flask entegrasyonu:
        from inference import EmotionPredictor
        predictor = EmotionPredictor()

        @app.route('/predict', methods=['POST'])
        def predict():
            image = request.files['image']
            result = predictor.predict_from_bytes(image.read())
            return jsonify(result)
    """

    def __init__(self, model_path=None):
        """
        Tahmin sınıfını başlat ve modeli yükle.

        Parametreler:
            model_path (str): Model ağırlıkları dosya yolu
                            (varsayılan: config.BEST_MODEL_PATH)
        """
        if model_path is None:
            model_path = config.BEST_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {model_path}\n"
                "Önce 'python main.py --mode train' ile modeli eğitin."
            )

        # Modeli yükle
        print(f"[INFO] Model yükleniyor: {model_path}")
        self.model = get_model(pretrained_path=model_path)
        self.model.to(config.DEVICE)
        self.model.eval()  # Değerlendirme modu (dropout kapalı)
        print("[INFO] Model başarıyla yüklendi ve hazır.")

    def preprocess(self, image):
        """
        Görüntüyü model girişine uygun formata dönüştürür.

        İşlem adımları:
        1. Gri tonlamaya çevir (eğer renkli ise)
        2. 48x48 boyutuna yeniden boyutlandır
        3. [0, 255] → [0.0, 1.0] normalize et
        4. (1, 1, 48, 48) tensor'a dönüştür

        Parametreler:
            image: PIL Image veya numpy array

        Returns:
            torch.Tensor: [1, 1, 48, 48] boyutunda tensor
        """
        # Numpy array ise PIL Image'e dönüştür
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Gri tonlamaya çevir
        if image.mode != 'L':
            image = image.convert('L')

        # 48x48'e yeniden boyutlandır
        image = image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)

        # Numpy array'e dönüştür ve normalize et
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Tensor'a dönüştür: (48, 48) → (1, 1, 48, 48) → [batch, channel, H, W]
        tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

        return tensor.to(config.DEVICE)

    @torch.no_grad()
    def predict(self, tensor):
        """
        Preprocessed tensor üzerinde tahmin yapar.

        Parametreler:
            tensor (Tensor): [1, 1, 48, 48] boyutunda giriş

        Returns:
            dict: {
                'emotion': str,          # Tahmin edilen duygu
                'emotion_id': int,       # Duygu ID'si (0-6)
                'confidence': float,     # Güven skoru (0-1)
                'probabilities': dict    # Her duygu için olasılık
            }
        """
        # Model çıkışı (logitler)
        outputs = self.model(tensor)

        # Softmax ile olasılıklara dönüştür
        probabilities = F.softmax(outputs, dim=1)

        # En yüksek olasılıklı sınıf
        confidence, predicted = torch.max(probabilities, 1)

        emotion_id = predicted.item()
        emotion_name = config.EMOTION_LABELS[emotion_id]
        conf = confidence.item()

        # Her sınıf için olasılıklar
        probs = probabilities[0].cpu().numpy()
        prob_dict = {
            config.EMOTION_LABELS[i]: round(float(probs[i]), 4)
            for i in range(config.NUM_CLASSES)
        }

        return {
            'emotion': emotion_name,
            'emotion_id': emotion_id,
            'confidence': round(conf, 4),
            'probabilities': prob_dict
        }

    def predict_from_image(self, image_path):
        """
        Dosya yolundan duygu tahmini yapar.

        Parametreler:
            image_path (str): Görüntü dosyası yolu

        Returns:
            dict: Tahmin sonucu
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

        image = Image.open(image_path)
        tensor = self.preprocess(image)
        return self.predict(tensor)

    def predict_from_array(self, numpy_array):
        """
        Numpy array'den duygu tahmini yapar.
        OpenCV gibi kütüphanelerle yakalanan frame'ler için uygundur.

        Parametreler:
            numpy_array (np.ndarray): Görüntü array'i

        Returns:
            dict: Tahmin sonucu
        """
        tensor = self.preprocess(numpy_array)
        return self.predict(tensor)

    def predict_from_bytes(self, image_bytes):
        """
        Byte dizisinden duygu tahmini yapar.
        Web API'lerden gelen görüntüler için uygundur.

        Parametreler:
            image_bytes (bytes): Görüntü byte verisi

        Returns:
            dict: Tahmin sonucu
        """
        import io
        image = Image.open(io.BytesIO(image_bytes))
        tensor = self.preprocess(image)
        return self.predict(tensor)


def main():
    """Komut satırından tekil tahmin için."""
    import argparse

    parser = argparse.ArgumentParser(description="Duygu Tahmini")
    parser.add_argument("--image", type=str, required=True,
                        help="Tahmin yapılacak görüntü yolu")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dosya yolu (varsayılan: best_model.pth)")
    args = parser.parse_args()

    predictor = EmotionPredictor(args.model)
    result = predictor.predict_from_image(args.image)

    print(f"\n{'=' * 40}")
    print(f"  Tahmin Sonucu")
    print(f"{'=' * 40}")
    print(f"  Görüntü:  {args.image}")
    print(f"  Duygu:    {result['emotion']}")
    print(f"  Güven:    {result['confidence'] * 100:.1f}%")
    print(f"\n  Tüm Olasılıklar:")
    for emotion, prob in sorted(result['probabilities'].items(),
                                 key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {emotion:<12} {prob * 100:5.1f}% {bar}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
