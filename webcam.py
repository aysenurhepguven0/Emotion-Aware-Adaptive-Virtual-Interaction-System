"""
webcam.py - Real-time Webcam Emotion Recognition
================================================
OpenCV + MTCNN ile webcam'den yüz algılama ve duygu tahmini.
After training the model, you can test live with webcam.

Usage:
    python webcam.py
    python webcam.py --model outputs/models/best_model.pth

Controls:
    q or ESC  -> Exit
    s           -> Save screenshot

Requirements:
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
# Emotion class colors (BGR format - OpenCV)
# ============================================================
EMOTION_COLORS = {
    0: (0, 0, 255),       # Angry    -> Kırmızı
    1: (0, 255, 255),     # Happy    -> Sarı
    2: (255, 0, 0),       # Sad      -> Mavi
    3: (0, 165, 255),     # Surprise -> Turuncu
    4: (200, 200, 200),   # Neutral  -> Gri
}

# Turkish emotion names
EMOTION_TURKISH = {
    0: "Kizgin",
    1: "Mutlu",
    2: "Uzgun",
    3: "Saskin",
    4: "Notr",
}


class WebcamEmotionDetector:
    """
    Real-time facial expression recognition via webcam.

    Processing steps:
    1. Capture frame from webcam
    2. Detect faces with MTCNN
    3. Crop face region, convert to 48x48 grayscale
    4. Predict emotion with Mini-Xception model
    5. Display result on screen
    """

    def __init__(self, model_path=None):
        """
        Initialize detector.

        Args:
            model_path (str): Model weights dosya yolu
        """
        if model_path is None:
            model_path = config.BEST_MODEL_PATH

        # ---- Model yükle ----
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "First train the model with 'python main.py --mode train'."
            )

        print("[INFO] Loading model...")
        self.model = get_model(pretrained_path=model_path)
        self.model.to(config.DEVICE)
        self.model.eval()

        # ---- Load MTCNN for face detection ----
        from facenet_pytorch import MTCNN as FaceDetector
        self.face_detector = FaceDetector(
            keep_all=True,
            device=config.DEVICE,
            min_face_size=48,
            thresholds=[0.6, 0.7, 0.7]
        )
        print("[INFO] MTCNN face detector loaded.")
        print("[INFO] System ready!\n")

    def preprocess_face(self, face_roi):
        """
        Preprocesses face region for model input.

        Args:
            face_roi (numpy.ndarray): Cropped face region (BGR or Gray)

        Returns:
            torch.Tensor: [1, 1, 48, 48] model input
        """
        # Convert to grayscale (if color)
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi

        # Resize to 48x48
        resized = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE),
                             interpolation=cv2.INTER_AREA)

        # Normalize [0, 255] -> [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to tensor: (48,48) -> (1, 1, 48, 48)
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(config.DEVICE)

    @torch.no_grad()
    def predict_emotion(self, face_tensor):
        """
        Predicts emotion from face tensor.

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
        Draws face bounding box and emotion info on frame.

        Args:
            frame: Original frame
            faces: Detected face coordinates
            predictions: Prediction results for each face
        """
        for (x, y, w, h), (emo_id, emo_name, conf, probs) in zip(faces, predictions):
            color = EMOTION_COLORS.get(emo_id, (255, 255, 255))
            turkish_name = EMOTION_TURKISH.get(emo_id, emo_name)

            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw emotion label (above bounding box)
            label = f"{emo_name} ({turkish_name}) {conf*100:.0f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            # Label background
            cv2.rectangle(frame,
                          (x, y - label_size[1] - 12),
                          (x + label_size[0] + 8, y),
                          color, -1)

            # Label text
            cv2.putText(frame, label,
                        (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Probability bars (on the right side)
            bar_x = x + w + 10
            bar_y_start = y
            bar_height = 16
            bar_max_width = 100

            for i in range(config.NUM_CLASSES):
                prob = probs[i]
                bar_w = int(prob * bar_max_width)
                by = bar_y_start + i * (bar_height + 4)

                # Bar background
                cv2.rectangle(frame,
                              (bar_x, by),
                              (bar_x + bar_max_width, by + bar_height),
                              (50, 50, 50), -1)

                # Bar fill
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
        Starts the webcam loop.

        Args:
            camera_id (int): Camera ID (default: 0 = main webcam)
        """
        print("=" * 50)
        print("  WEBCAM EMOTION RECOGNITION")
        print("=" * 50)
        print("  Controls:")
        print("    q / ESC  -> Exit")
        print("    s        -> Save screenshot")
        print("=" * 50)

        # Open webcam
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("[HATA] Could not open webcam! Make sure camera is connected.")
            return

        # Set resolution (for performance)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_counter = 0
        fps_time = time.time()
        fps_display = 0

        print("\n[INFO] Webcam opened. Press 'q' to exit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[HATA] Could not read frame!")
                break

            # Mirror the frame horizontally (natural mirror view)
            frame = cv2.flip(frame, 1)

            # Detect faces with MTCNN (RGB formatı gerekli)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.face_detector.detect(frame_rgb)

            # Predict emotion for each face
            predictions = []
            faces = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    # Check bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    w = x2 - x1
                    h = y2 - y1
                    if w < 20 or h < 20:
                        continue

                    # Crop face region
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_roi = gray[y1:y2, x1:x2]

                    # Predict emotion
                    face_tensor = self.preprocess_face(face_roi)
                    prediction = self.predict_emotion(face_tensor)
                    predictions.append(prediction)
                    faces.append((x1, y1, w, h))

            # Draw results
            frame = self.draw_results(frame, faces, predictions)

            # Calculate and display FPS
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

            # Display window
            cv2.imshow("Duygu Tanima - FER2013 (q: Cikis)", frame)

            # Key control
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                print("\n[INFO] Exit yapılıyor...")
                break
            elif key == ord('s'):  # Save screenshot
                screenshot_path = os.path.join(
                    config.OUTPUT_DIR,
                    f"screenshot_{int(time.time())}.png"
                )
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot saved: {screenshot_path}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam closed.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Webcam Emotion Recognition")
    parser.add_argument("--model", type=str, default=None,
                        help="Model file path (default: best_model.pth)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (default: 0)")
    args = parser.parse_args()

    detector = WebcamEmotionDetector(args.model)
    detector.run(args.camera)


if __name__ == "__main__":
    main()
