# Emotion-Aware Adaptive Virtual Interaction System

**Duygu Farkındalıklı Uyarlanabilir Sanal Etkileşim Sistemi**

Gerçek zamanlı yüz ifadesi tanıma ve duyguya duyarlı sanal etkileşim sistemi. Webcam üzerinden yüz ifadelerini algılayarak 6 temel Ekman duygusunu sınıflandırır ve sonuçları Unity 3D ortamında **generative data sculpture** (üretken veri heykeli) olarak görselleştirir.

---

## Proje Özeti

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

## Sistem Mimarisi

```
+---------------------------------------------------+
|  Layer 1 -- Data Pipeline                         |
|  FER2013 / FER+ / RAF-DB / CK+ -> Augmentation    |
+---------------------------------------------------+
|  Layer 2 -- Candidate Model Architectures         |
|  Mini-Xception / EfficientNet-B0 / ResNet-18      |
|  HSEmotion (AffectNet pre-trained)                |
+---------------------------------------------------+
|  Layer 3 -- Training & Regularisation             |
|  Adam / Weighted Loss / LR Scheduler / Early Stop |
+---------------------------------------------------+
|  Layer 4 -- Real-time Application                 |
|  Webcam + MTCNN --> Visualisation (OpenCV)         |
|                 --> TCP Server --> Unity 3D        |
+---------------------------------------------------+
```

---

## Proje Yapısı

```
proje_kodu/
|-- config.py                 # Tum konfigurasyon ve hyperparametreler
|-- main.py                   # Ana calistirma scripti
|-- train.py                  # Egitim pipeline'i
|-- evaluate.py               # Model degerlendirme ve metrikler
|-- compare_models.py         # Model karsilastirma scripti
|-- webcam.py                 # Gercek zamanli webcam modu (MTCNN)
|-- unity_bridge.py           # Python <-> Unity TCP koprusu
|-- inference.py              # Tekil goruntu tahmini
|-- colab_train.ipynb         # Google Colab egitim notebook'u
|-- requirements.txt          # Python bagimliliklari
|
|-- data/
|   |-- dataset.py            # Dataset siniflari ve DataLoader'lar
|   |-- fer2013/              # FER2013 veri seti (train/test)
|   |-- ferplus/              # FER+ veri seti (train/validation/test)
|   |-- raf-db/               # RAF-DB veri seti
|   +-- ck+/                  # CK+ veri seti
|
|-- models/
|   |-- mini_xception.py      # Mini-Xception CNN mimarisi
|   |-- efficientnet.py       # EfficientNet-B0 (transfer learning)
|   |-- resnet.py             # ResNet-18 (transfer learning)
|   +-- hsemotion_model.py    # HSEmotion (AffectNet pre-trained)
|
|-- utils/
|   +-- visualization.py      # Grafik ve gorsellestirme araclari
|
|-- unity/
|   |-- EmotionReceiver.cs    # Unity TCP alici script
|   +-- EmotionParticleSystem.cs  # Duygu tabanli parcacik heykeli
|
+-- outputs/
    |-- models/               # Egitilmis model agirliklari (.pth)
    +-- plots/                # Egitim grafikleri
```

---

## Kurulum

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
|-- train/
|   |-- angry/
|   |-- disgust/
|   |-- fear/
|   |-- happy/
|   |-- sad/
|   +-- surprise/
+-- test/
    |-- angry/
    +-- ...
```

---

## Kullanım

### Model Egitimi
```bash
# Mini-Xception ile egitim (varsayilan)
python train.py --dataset fer2013 --model mini_xception

# EfficientNet-B0 ile egitim (transfer learning)
python train.py --dataset fer2013 --model efficientnet

# ResNet-18 ile egitim
python train.py --dataset ferplus --model resnet

# HSEmotion ile egitim
python train.py --dataset ferplus --model hsemotion
```

### Model Karsilastirma
```bash
# Tum modelleri karsilastir
python compare_models.py --dataset ferplus
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

## Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
|-----------|---------------|
| **Python 3.9+** | Ana programlama dili |
| **PyTorch** | Derin öğrenme framework'ü |
| **MTCNN** | Gerçek zamanlı yüz algılama |
| **OpenCV** | Görüntü işleme ve webcam |
| **Unity 3D** | Generative data sculpture |
| **C#** | Unity scripting |

## Veri Setleri

| Dataset | Görüntü Sayısı | Boyut | Ortam |
|---------|---------------|-------|-------|
| **FER2013** | ~35,887 | 48x48 gri | Dogal |
| **FER+** | ~78,000 | 48x48 gri | Dogal |
| **RAF-DB** | ~29,672 | 100x100 RGB | Dogal |
| **CK+** | ~593 sekans | 640x490 gri | Laboratuvar |

---

## Lisans

Bu proje, Galatasaray Üniversitesi Bilgisayar Mühendisliği Bölümü bitirme projesi kapsamında geliştirilmiştir.

## Geliştirici

**Ayşenur Hepgüven**
Galatasaray Üniversitesi - Bilgisayar Mühendisliği
