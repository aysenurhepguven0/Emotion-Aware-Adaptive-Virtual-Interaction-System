# Emotion-Aware Adaptive Virtual Interaction System

**Duygu Farkındalıklı Uyarlanabilir Sanal Etkileşim Sistemi**

Gerçek zamanlı yüz ifadesi tanıma ve duyguya duyarlı sanal etkileşim sistemi. Webcam üzerinden yüz ifadelerini algılayarak 6 temel Ekman duygusunu sınıflandırır ve sonuçları Unity 3D ortamında **generative data sculpture** (üretken veri heykeli) olarak görselleştirir.

---

## 🎯 Proje Özeti

Bu proje, CNN tabanlı duygu tanıma modellerini karşılaştırmalı olarak değerlendirmekte ve en iyi modeli iki farklı modda dağıtmaktadır:

- **Webcam Modu**: MTCNN ile gerçek zamanlı yüz algılama + duygu tahmini
- **Unity 3D Modu**: TCP üzerinden iletilen duygu verileriyle yönlendirilen interaktif parçacık heykeli

### Hedef Duygular (6 Ekman)
| ID | Duygu | Emotion |
|----|-------|---------|
| 0 | Kızgın | Angry |
| 1 | Tiksinme | Disgust |
| 2 | Korku | Fear |
| 3 | Mutlu | Happy |
| 4 | Üzgün | Sad |
| 5 | Şaşkın | Surprise |

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────┐
│  Layer 1 — Data Pipeline                        │
│  FER2013 • RAF-DB • CK+ → Augmentation → Loader│
├─────────────────────────────────────────────────┤
│  Layer 2 — Candidate Model Architectures        │
│  Mini-Xception • EfficientNet-B0 • ResNet-18    │
├─────────────────────────────────────────────────┤
│  Layer 3 — Training & Regularisation            │
│  Adam • Weighted Loss • LR Scheduler • Early Stop│
├─────────────────────────────────────────────────┤
│  Layer 4 — Real-time Application                │
│  Webcam + MTCNN ──→ Visualisation (OpenCV)      │
│                 └──→ TCP Server → Unity 3D      │
└─────────────────────────────────────────────────┘
```

---

## 📁 Proje Yapısı

```
proje_kodu/
├── config.py                 # Tüm konfigürasyon ve hyperparametreler
├── main.py                   # Ana çalıştırma scripti
├── train.py                  # Eğitim pipeline'ı
├── evaluate.py               # Model değerlendirme ve metrikler
├── webcam.py                 # Gerçek zamanlı webcam modu (MTCNN)
├── unity_bridge.py           # Python ↔ Unity TCP köprüsü
├── inference.py              # Tekil görüntü tahmini
├── requirements.txt          # Python bağımlılıkları
│
├── data/
│   ├── dataset.py            # Dataset sınıfları ve DataLoader'lar
│   ├── fer2013/              # FER2013 veri seti (train/test)
│   ├── raf-db/               # RAF-DB veri seti
│   └── ck+/                  # CK+ veri seti
│
├── models/
│   ├── mini_xception.py      # Mini-Xception CNN mimarisi
│   └── efficientnet.py       # EfficientNet-B0 (transfer learning)
│
├── utils/
│   └── visualization.py      # Grafik ve görselleştirme araçları
│
├── unity/
│   ├── EmotionReceiver.cs    # Unity TCP alıcı script
│   └── EmotionParticleSystem.cs  # Duygu tabanlı parçacık heykeli
│
└── outputs/
    ├── models/               # Eğitilmiş model ağırlıkları (.pth)
    └── plots/                # Eğitim grafikleri
```

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.9+
- CUDA destekli GPU (opsiyonel, CPU'da da çalışır)

### 1. Depoyu klonlayın
```bash
git clone https://github.com/aysenurhepguven0/Emotion-Aware-Adaptive-Virtual-Interaction-System.git
cd Emotion-Aware-Adaptive-Virtual-Interaction-System
```

### 2. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 3. Veri setlerini hazırlayın
FER2013 veri setini `data/fer2013/` dizinine klasör formatında yerleştirin:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    └── ...
```

---

## 🚀 Kullanım

### Model Eğitimi
```bash
# Mini-Xception ile eğitim (varsayılan)
python train.py --dataset fer2013 --model mini_xception

# EfficientNet-B0 ile eğitim (transfer learning)
python train.py --dataset fer2013 --model efficientnet
```

### Model Değerlendirme
```bash
python evaluate.py --dataset fer2013
```

### Webcam ile Gerçek Zamanlı Tanıma
```bash
python webcam.py
```
- `q` veya `ESC`: Çıkış
- `s`: Ekran görüntüsü kaydet

### Unity 3D Modu
```bash
# 1. Python TCP sunucusunu başlat
python unity_bridge.py

# 2. Unity projesinde Play'e bas
```

### Tüm İşlemleri Sırayla Çalıştır
```bash
python main.py --mode all
```

---

## 🧠 Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
|-----------|---------------|
| **Python 3.9+** | Ana programlama dili |
| **PyTorch** | Derin öğrenme framework'ü |
| **MTCNN** | Gerçek zamanlı yüz algılama |
| **OpenCV** | Görüntü işleme ve webcam |
| **Unity 3D** | Generative data sculpture |
| **C#** | Unity scripting |

## 📊 Veri Setleri

| Dataset | Görüntü Sayısı | Boyut | Ortam |
|---------|---------------|-------|-------|
| **FER2013** | ~35,887 | 48×48 gri | Doğal |
| **RAF-DB** | ~29,672 | 100×100 RGB | Doğal |
| **CK+** | ~593 sekans | 640×490 gri | Laboratuvar |

---

## 📄 Lisans

Bu proje, Galatasaray Üniversitesi Bilgisayar Mühendisliği Bölümü bitirme projesi kapsamında geliştirilmiştir.

## 👩‍💻 Geliştirici

**Ayşenur Hepgüven**
Galatasaray Üniversitesi - Bilgisayar Mühendisliği
