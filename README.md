# 🔊 DeepFake-audio-Detection-using-TSSD-with-Attention-Mechanism

As deepfake technologies evolve, synthetic voice manipulation has become a significant threat—enabling voice impersonation, misinformation, and audio-based fraud. This project addresses these risks by implementing a **robust deep learning system** that accurately distinguishes real from fake audio using a **Time-Domain Synthetic Speech Detection (TSSD)** model enhanced with a **multi-head attention mechanism**.


## 📌 Overview

We built a high-performance audio classification model using a **1D CNN-based TSSD architecture** coupled with **multi-head self-attention** to extract and emphasize important temporal and spectral features. The system supports **real-time detection** via a Flask-powered web interface where users can upload `.wav` files and receive instant predictions.


## 🎯 Objective

To develop a scalable and real-time **deepfake audio detection** framework using time-series neural networks and **attention-based learning**, achieving high accuracy and robustness across diverse synthetic speech sources.


## ✨ Features

- **TSSD Architecture**: Captures temporal dependencies in audio using 1D CNN and residual blocks.
- **Attention Mechanism**: Multi-head self-attention layer enhances focus on critical segments of speech.
- **Rich Feature Extraction**: Uses MFCC, Mel Spectrogram, Chroma, ZCR, Spectral Centroid & Flatness.
- **High Performance**:  
  - ✅ Accuracy: **98.20%**  
  - ✅ EER: **0.0505**  
  - ✅ T-DCF: **0.0044**
- **Web Interface**: Real-time prediction with audio upload and user login through Flask.

## 🛠️ Tech Stack

- **Python** – Core programming language
- **PyTorch** – for model building and training
- **NumPy, Librosa** – Audio preprocessing and feature extraction
- **Flask** – Web framework for UI and backend integration
- **HTML** – Frontend integration

## 📂 Dataset

- **Real Audio Samples**: 15,000 clips sourced from public speech datasets.
- **Fake Audio Samples**: 15,000 clips generated using TTS models like WaveNet.
- Format: `.wav` files, processed into 2–3 second segments.

## 🧠 Model Architecture

- **Input**: Audio signals
- **Feature Extraction**: MFCC, Mel Spectrogram, Chroma, ZCR, Spectral Centroid, Spectral Flatness
- **TSSD Layers**: 1D CNN with Residual Blocks
- **Attention Layer**: Multi-head attention mechanism
- **Dense Layers**: Fully connected layers
- **Output**: Softmax layer classifying audio as Real or Fake
![Model Architecture](WhatsApp Image 2025-06-09 at 10.35.49 PM)


## 📊 Performance Metrics

| Model                | Accuracy | EER    | T-DCF   |
|---------------------|----------|--------|---------|
| TSSD + Attention    | 98.2%    | 0.0505 | 0.0044  |
| TSSD                | 94.5%    | 0.0905 | 0.0244  |
| Multilayer Perceptron | 88.0%  |   -    |   -     |

- **AUC Score**: 0.99  
- Excellent generalization across real and fake voice samples

## 💻 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-audio-detection-tssd.git
cd deepfake-audio-detection-tssd

# Install dependencies
pip install -r requirements.txt

# Run the Flask web app
python app.py



