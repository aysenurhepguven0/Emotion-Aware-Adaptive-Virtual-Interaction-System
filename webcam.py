"""
webcam.py - Gerçek Zamanlı Webcam Duygu Tanıma
================================================
OpenCV + MTCNN ile webcam'den yüz algılama ve duygu tahmini.
Modeli eğittikten sonra webcam ile canlı test yapabilirsiniz.

Kullanım:
    python webcam.py
    python webcam.py --model outputs/models/best_model.pth

Kontroller:
    q veya ESC  → Çıkış
    s           → Ekran görüntüsü kaydet

Gereksinimler:
    pip install opencv-python facenet-pytorch
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.mini_xception import get_model


# ============================================================
# Duygu sınıfları için renkler (BGR formatı - OpenCV)
# ============================================================
EMOTION_COLORS = {
    0: (0, 0, 255),       # Angry    → Kırmızı
    1: (0, 128, 0),       # Disgust  → Koyu Yeşil
    2: (128, 0, 128),     # Fear     → Mor
    3: (0, 255, 255),     # Happy    → Sarı
    4: (255, 0, 0),       # Sad      → Mavi
    5: (0, 165, 255),     # Surprise → Turuncu
}

# Türkçe duygu isimleri
EMOTION_TURKISH = {
    0: "Kizgin",
    1: "Tiksinme",
    2: "Korku",
    3: "Mutlu",
    4: "Uzgun",
    5: "Saskin",
}


class WebcamEmotionDetector:
    """
    Webcam ile gerçek zamanlı yüz ifadesi tanıma.

    İşlem adımları:
    1. Webcam'den frame yakala
    2. MTCNN ile yüz algıla
    3. Yüz bölgesini kırp, 48x48 gri tonlamaya dönüştür
    4. Mini-Xception modeli ile duygu tahmini yap
    5. Sonucu ekranda göster
    """

    def __init__(self, model_path=None):
        """
        Detector'ı başlat.

        Parametreler:
            model_path (str): Model ağırlıkları dosya yolu
        """
        if model_path is None:
            model_path = config.BEST_MODEL_PATH

        # ---- Model yükle ----
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {model_path}\n"
                "Önce 'python main.py --mode train' ile modeli eğitin."
            )

        print("[INFO] Model yükleniyor...")
        self.model = get_model(pretrained_path=model_path)
        self.model.to(config.DEVICE)
        self.model.eval()

        # ---- Yüz algılama için MTCNN yükle ----
        from facenet_pytorch import MTCNN as FaceDetector
        self.face_detector = FaceDetector(
            keep_all=True,
            device=config.DEVICE,
            min_face_size=48,
            thresholds=[0.6, 0.7, 0.7]
        )
        print("[INFO] MTCNN yüz algılayıcı yüklendi.")
        print("[INFO] Sistem hazır!\n")

    def preprocess_face(self, face_roi):
        """
        Yüz bölgesini model girişine uygun hale getirir.

        Parametreler:
            face_roi (numpy.ndarray): Kırpılmış yüz bölgesi (BGR veya Gri)

        Returns:
            torch.Tensor: [1, 1, 48, 48] model girişi
        """
        # Gri tonlamaya çevir (renkli ise)
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi

        # 48x48'e yeniden boyutlandır
        resized = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE),
                             interpolation=cv2.INTER_AREA)

        # Normalize et [0, 255] → [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # Tensor'a dönüştür: (48,48) → (1, 1, 48, 48)
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(config.DEVICE)

    @torch.no_grad()
    def predict_emotion(self, face_tensor):
        """
        Yüz tensöründen duygu tahmini yapar.

        Returns:
            tuple: (emotion_id, emotion_name, confidence, probabilities)
        """
        outputs = self.model(face_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        emotion_id = predicted.item()
        emotion_name = config.EMOTION_LABELS[emotion_id]
        conf = confidence.item()
        probs = probabilities[0].cpu().numpy()

        return emotion_id, emotion_name, conf, probs

    def draw_results(self, frame, faces, predictions):
        """
        Frame üzerine yüz kutucuğu ve duygu bilgisi çizer.

        Parametreler:
            frame: Orijinal frame
            faces: Algılanan yüzlerin koordinatları
            predictions: Her yüz için tahmin sonuçları
        """
        for (x, y, w, h), (emo_id, emo_name, conf, probs) in zip(faces, predictions):
            color = EMOTION_COLORS.get(emo_id, (255, 255, 255))
            turkish_name = EMOTION_TURKISH.get(emo_id, emo_name)

            # Yüz çerçevesi çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Duygu etiketi çiz (çerçevenin üstüne)
            label = f"{emo_name} ({turkish_name}) {conf*100:.0f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            # Etiket arka planı
            cv2.rectangle(frame,
                          (x, y - label_size[1] - 12),
                          (x + label_size[0] + 8, y),
                          color, -1)

            # Etiket metni
            cv2.putText(frame, label,
                        (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Olasılık çubukları (sağ tarafta)
            bar_x = x + w + 10
            bar_y_start = y
            bar_height = 16
            bar_max_width = 100

            for i in range(config.NUM_CLASSES):
                prob = probs[i]
                bar_w = int(prob * bar_max_width)
                by = bar_y_start + i * (bar_height + 4)

                # Bar arka planı
                cv2.rectangle(frame,
                              (bar_x, by),
                              (bar_x + bar_max_width, by + bar_height),
                              (50, 50, 50), -1)

                # Bar dolgusu
                bar_color = EMOTION_COLORS.get(i, (200, 200, 200))
                cv2.rectangle(frame,
                              (bar_x, by),
                              (bar_x + bar_w, by + bar_height),
                              bar_color, -1)

                # Label
                bar_label = f"{config.EMOTION_LABELS[i][:3]} {prob*100:.0f}%"
                cv2.putText(frame, bar_label,
                            (bar_x + bar_max_width + 5, by + bar_height - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (200, 200, 200), 1)

        return frame

    def run(self, camera_id=0):
        """
        Webcam döngüsünü başlatır.

        Parametreler:
            camera_id (int): Kamera ID'si (varsayılan: 0 = ana webcam)
        """
        print("=" * 50)
        print("  WEBCAM DUYGU TANIMA")
        print("=" * 50)
        print("  Kontroller:")
        print("    q / ESC  → Çıkış")
        print("    s        → Ekran görüntüsü kaydet")
        print("=" * 50)

        # Webcam'i aç
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("[HATA] Webcam açılamadı! Kamera bağlı olduğundan emin olun.")
            return

        # Çözünürlük ayarla (performans için)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_counter = 0
        fps_time = time.time()
        fps_display = 0

        print("\n[INFO] Webcam açıldı. 'q' tuşuna basarak çıkabilirsiniz.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[HATA] Frame okunamadı!")
                break

            # MTCNN ile yüz algıla (RGB formatı gerekli)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.face_detector.detect(frame_rgb)

            # Her yüz için duygu tahmini
            predictions = []
            faces = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    # Sınırları kontrol et
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    w = x2 - x1
                    h = y2 - y1
                    if w < 20 or h < 20:
                        continue

                    # Yüz bölgesini kırp
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_roi = gray[y1:y2, x1:x2]

                    # Duygu tahmini yap
                    face_tensor = self.preprocess_face(face_roi)
                    prediction = self.predict_emotion(face_tensor)
                    predictions.append(prediction)
                    faces.append((x1, y1, w, h))

            # Sonuçları çiz
            frame = self.draw_results(frame, faces, predictions)

            # FPS hesapla ve göster
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.time()

            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Yuz sayisi: {len(faces)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)

            # Pencereyi göster
            cv2.imshow("Duygu Tanima - FER2013 (q: Cikis)", frame)

            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q veya ESC
                print("\n[INFO] Çıkış yapılıyor...")
                break
            elif key == ord('s'):  # Screenshot kaydet
                screenshot_path = os.path.join(
                    config.OUTPUT_DIR,
                    f"screenshot_{int(time.time())}.png"
                )
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Ekran görüntüsü kaydedildi: {screenshot_path}")

        # Temizle
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam kapatıldı.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Webcam Duygu Tanıma")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dosya yolu (varsayılan: best_model.pth)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Kamera ID (varsayılan: 0)")
    args = parser.parse_args()

    detector = WebcamEmotionDetector(args.model)
    detector.run(args.camera)


if __name__ == "__main__":
    main()
