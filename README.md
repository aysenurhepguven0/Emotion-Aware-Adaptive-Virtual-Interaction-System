# Emotion-Aware Adaptive Virtual Interaction System

A real-time facial expression recognition and emotion-aware virtual interaction system. It detects facial expressions through a webcam, classifies them into 5 emotion categories, and visualizes the results as a **generative data sculpture** in Unity 3D.

---

## Project Overview

This project comparatively evaluates CNN-based emotion recognition models and deploys the best-performing model in two different modes:

- **Webcam Mode**: Real-time face detection with MTCNN + emotion prediction
- **Unity 3D Mode**: Interactive particle sculpture driven by emotion data transmitted over TCP

### Target Emotions (5 Classes)

| ID | Emotion  |
|----|----------|
| 0  | Angry    |
| 1  | Happy    |
| 2  | Sad      |
| 3  | Surprise |
| 4  | Neutral  |

---

## System Architecture

```
+---------------------------------------------------+
|  Layer 1 -- Data Pipeline                         |
|  FER2013 / FER+ / RAF-DB / CK+ -> Augmentation   |
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

## Project Structure

```
proje_kodu/
|-- config.py                 # All configurations and hyperparameters
|-- main.py                   # Main execution script
|-- train.py                  # Training pipeline
|-- evaluate.py               # Model evaluation and metrics
|-- compare_models.py         # Model comparison script
|-- webcam.py                 # Real-time webcam mode (MTCNN)
|-- unity_bridge.py           # Python <-> Unity TCP bridge
|-- inference.py              # Single image prediction
|-- colab_train.ipynb         # Google Colab training notebook
|-- requirements.txt          # Python dependencies
|
|-- data/
|   |-- dataset.py            # Dataset classes and DataLoaders
|   |-- fer2013/              # FER2013 dataset (train/test)
|   |-- ferplus/              # FER+ dataset (train/validation/test)
|   |-- raf-db/               # RAF-DB dataset
|   +-- ck+/                  # CK+ dataset
|
|-- models/
|   |-- mini_xception.py      # Mini-Xception CNN architecture
|   |-- efficientnet.py       # EfficientNet-B0 (transfer learning)
|   |-- resnet.py             # ResNet-18 (transfer learning)
|   +-- hsemotion_model.py    # HSEmotion (AffectNet pre-trained)
|
|-- utils/
|   +-- visualization.py      # Plotting and visualization utilities
|
|-- unity/
|   |-- EmotionReceiver.cs    # Unity TCP receiver script
|   +-- EmotionParticleSystem.cs  # Emotion-based particle sculpture
|
+-- outputs/
    |-- models/               # Trained model weights (.pth)
    +-- plots/                # Training plots
```

---

## Installation

### Requirements
- Python 3.9+
- CUDA-enabled GPU (optional, also runs on CPU)

### 1. Clone the repository
```bash
git clone https://github.com/aysenurhepguven0/Emotion-Aware-Adaptive-Virtual-Interaction-System.git
cd Emotion-Aware-Adaptive-Virtual-Interaction-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare datasets
Place the FER2013 dataset in the `data/fer2013/` directory in folder format:
```
data/fer2013/
|-- train/
|   |-- angry/
|   |-- happy/
|   |-- sad/
|   +-- surprise/
+-- test/
    |-- angry/
    +-- ...
```

---

## Usage

### Model Training
```bash
# Train with Mini-Xception (default)
python train.py --dataset fer2013 --model mini_xception

# Train with EfficientNet-B0 (transfer learning)
python train.py --dataset fer2013 --model efficientnet

# Train with ResNet-18
python train.py --dataset ferplus --model resnet

# Train with HSEmotion
python train.py --dataset ferplus --model hsemotion
```

### Model Comparison
```bash
# Compare all models
python compare_models.py --dataset ferplus
```

### Model Evaluation
```bash
python evaluate.py --dataset fer2013
```

### Real-time Recognition via Webcam
```bash
python webcam.py
```
- `q` or `ESC`: Quit
- `s`: Save screenshot

### Unity 3D Mode
```bash
# 1. Start the Python TCP server
python unity_bridge.py

# 2. Press Play in the Unity project
```

### Run All Steps Sequentially
```bash
python main.py --mode all
```

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Primary programming language |
| **PyTorch** | Deep learning framework |
| **MTCNN** | Real-time face detection |
| **OpenCV** | Image processing and webcam |
| **Unity 3D** | Generative data sculpture |
| **C#** | Unity scripting |

## Datasets

| Dataset | Image Count | Size | Environment |
|---------|------------|------|-------------|
| **FER2013** | ~35,887 | 48x48 grayscale | Natural |
| **FER+** | ~78,000 | 48x48 grayscale | Natural |
| **RAF-DB** | ~29,672 | 100x100 RGB | Natural |
| **CK+** | ~593 sequences | 640x490 grayscale | Laboratory |

---

## License

This project was developed as a senior capstone project at Galatasaray University, Department of Computer Engineering.

## Author

**Aysenur Hepguven**
Galatasaray University - Computer Engineering
