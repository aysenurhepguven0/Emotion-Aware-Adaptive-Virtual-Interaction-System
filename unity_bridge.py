"""
unity_bridge.py - Python <-> Unity TCP Köprüsü
==============================================
Webcam'den duygu tanıma yaparak sonuçları TCP üzerinden
Unity'ye gerçek zamanlı gönderir.

Kullanım:
    python unity_bridge.py
    python unity_bridge.py --port 5555 --camera 0

Unity tarafında EmotionReceiver.cs ile bağlanılır.
"""

import os
import sys
import time
import json
import socket
import threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.mini_xception import get_model


# ============================================================
# TCP Socket Sunucusu (Ayrı Thread)
# ============================================================
class EmotionServer:
    """
    TCP sunucusu: Unity client'lara duygu verisini JSON olarak gönderir.
    Birden fazla client aynı anda bağlanabilir.
    """

    def __init__(self, host="0.0.0.0", port=5555):
        self.host = host
        self.port = port
        self.clients = []
        self.clients_lock = threading.Lock()
        self.running = False
        self.server_socket = None

    def start(self):
        """Sunucuyu ayrı bir thread'de başlat."""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # accept() timeout

        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()
        print(f"[SERVER] TCP sunucu başlatıldı: {self.host}:{self.port}")
        print(f"[SERVER] Unity bağlantısı bekleniyor...")

    def _accept_loop(self):
        """Yeni client bağlantılarını kabul et."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                with self.clients_lock:
                    self.clients.append(client_socket)
                print(f"[SERVER] Unity bağlandı: {addr}")
            except socket.timeout:
                continue
            except OSError:
                break

    def send_emotion(self, data):
        """
        Tüm bağlı client'lara duygu verisini gönder.

        Parametreler:
            data (dict): Duygu verisi dict'i
        """
        message = json.dumps(data) + "\n"
        message_bytes = message.encode("utf-8")

        disconnected = []
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.sendall(message_bytes)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    disconnected.append(client)

            for client in disconnected:
                self.clients.remove(client)
                try:
                    client.close()
                except:
                    pass
                print("[SERVER] Bir Unity client bağlantısı koptu.")

    @property
    def client_count(self):
        with self.clients_lock:
            return len(self.clients)

    def stop(self):
        """Sunucuyu kapat."""
        self.running = False
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("[SERVER] TCP sunucu kapatıldı.")


# ============================================================
# Webcam + Duygu Tanıma + Unity Köprüsü
# ============================================================
class UnityEmotionBridge:
    """
    Webcam'den yüz algılayıp duygu tahmini yapar,
    sonuçları hem ekranda gösterir hem TCP ile Unity'ye gönderir.
    """

    EMOTION_COLORS = {
        0: (0, 0, 255),       # Angry    -> Kırmızı
        1: (0, 255, 255),     # Happy    -> Sarı
        2: (255, 0, 0),       # Sad      -> Mavi
        3: (0, 165, 255),     # Surprise -> Turuncu
        4: (200, 200, 200),   # Neutral  -> Gri
    }

    def __init__(self, model_path=None, port=5555):
        # ---- Model yükle ----
        if model_path is None:
            model_path = config.BEST_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {model_path}\n"
                "Önce 'python main.py --mode train' ile modeli eğitin."
            )

        print("[BRIDGE] Model yükleniyor...")
        self.model = get_model(pretrained_path=model_path)
        self.model.to(config.DEVICE)
        self.model.eval()

        # ---- Haar Cascade ----
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # ---- TCP Sunucu ----
        self.server = EmotionServer(port=port)

        print("[BRIDGE] Sistem hazır!\n")

    def preprocess_face(self, face_roi):
        """Yüz bölgesini 48x48 tensöre dönüştür."""
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi

        resized = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(config.DEVICE)

    @torch.no_grad()
    def predict_emotion(self, face_tensor):
        """Duygu tahmini yap, dict olarak döndür."""
        outputs = self.model(face_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        emotion_id = predicted.item()
        emotion_name = config.EMOTION_LABELS[emotion_id]
        conf = confidence.item()
        probs = probabilities[0].cpu().numpy()

        prob_dict = {
            config.EMOTION_LABELS[i]: round(float(probs[i]), 4)
            for i in range(config.NUM_CLASSES)
        }

        return {
            "emotion": emotion_name,
            "emotion_id": emotion_id,
            "confidence": round(conf, 4),
            "probabilities": prob_dict
        }

    def run(self, camera_id=0, show_preview=True):
        """
        Ana döngü: Webcam -> Duygu Tahmini -> Unity'ye Gönder

        Parametreler:
            camera_id (int): Kamera ID'si
            show_preview (bool): OpenCV önizleme penceresi göster
        """

        # TCP sunucuyu başlat
        self.server.start()

        # Webcam'i aç
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[HATA] Webcam açılamadı!")
            self.server.stop()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_counter = 0
        fps_time = time.time()
        fps_display = 0

        print("=" * 55)
        print("  UNITY DUYGU KÖPRÜSÜ")
        print("=" * 55)
        print(f"  TCP Port : {self.server.port}")
        print(f"  Kamera   : {camera_id}")
        print(f"  Önizleme : {'Açık' if show_preview else 'Kapalı'}")
        print("  Kontrol  : q / ESC -> Çıkış")
        print("=" * 55)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Yüz algıla
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(48, 48),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # Her yüz için duygu tahmini
                all_predictions = []
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_tensor = self.preprocess_face(face_roi)
                    prediction = self.predict_emotion(face_tensor)
                    prediction["face_x"] = int(x)
                    prediction["face_y"] = int(y)
                    prediction["face_w"] = int(w)
                    prediction["face_h"] = int(h)
                    all_predictions.append(prediction)

                # Unity'ye gönder (en baskın yüz veya boş)
                if all_predictions:
                    # En yüksek confidence'a sahip yüzü gönder
                    best = max(all_predictions, key=lambda p: p["confidence"])
                    unity_data = {
                        "emotion": best["emotion"],
                        "emotion_id": best["emotion_id"],
                        "confidence": best["confidence"],
                        "probabilities": best["probabilities"],
                        "face_count": len(faces),
                        "timestamp": round(time.time(), 3)
                    }
                else:
                    # Yüz algılanmadı
                    unity_data = {
                        "emotion": "None",
                        "emotion_id": -1,
                        "confidence": 0.0,
                        "probabilities": {},
                        "face_count": 0,
                        "timestamp": round(time.time(), 3)
                    }

                # TCP ile gönder
                self.server.send_emotion(unity_data)

                # OpenCV önizleme
                if show_preview:
                    for (x, y, w, h), pred in zip(faces, all_predictions):
                        color = self.EMOTION_COLORS.get(pred["emotion_id"], (255, 255, 255))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                        label = f"{pred['emotion']} {pred['confidence']*100:.0f}%"
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # FPS
                    fps_counter += 1
                    if time.time() - fps_time >= 1.0:
                        fps_display = fps_counter / (time.time() - fps_time)
                        fps_counter = 0
                        fps_time = time.time()

                    status = f"FPS: {fps_display:.0f} | Yuz: {len(faces)} | Unity: {self.server.client_count}"
                    cv2.putText(frame, status, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.imshow("Unity Duygu Koprüsü (q: Cikis)", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                else:
                    # Başsız modda küçük bekleme
                    time.sleep(0.03)

        except KeyboardInterrupt:
            print("\n[BRIDGE] Ctrl+C algılandı.")
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            self.server.stop()
            print("[BRIDGE] Köprü kapatıldı.")


# ============================================================
# Komut satırı arayüzü
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Python -> Unity Duygu Tanıma Köprüsü"
    )
    parser.add_argument("--port", type=int, default=5555,
                        help="TCP port (varsayılan: 5555)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Kamera ID (varsayılan: 0)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dosya yolu")
    parser.add_argument("--no-preview", action="store_true",
                        help="OpenCV önizleme penceresini kapat (headless mod)")
    args = parser.parse_args()

    bridge = UnityEmotionBridge(model_path=args.model, port=args.port)
    bridge.run(camera_id=args.camera, show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
