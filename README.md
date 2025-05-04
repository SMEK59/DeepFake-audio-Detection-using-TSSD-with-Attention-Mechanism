# 🔊 DeepFake-audio-Detection-using-TSSD-with-Attention-Mechanism

A deep learning-based system for detecting synthetic (deepfake) audio using a Time-Domain Synthetic Speech Detection (TSSD) model enhanced with a multi-head attention mechanism. The system analyzes key spectral and temporal features such as MFCC, Mel Spectrogram, Chroma, Zero-Crossing Rate, Spectral Centroid, and Spectral Flatness to classify audio clips as real or fake with high accuracy. A Flask web interface allows users to upload audio files and receive real-time classification results.

## 📌 Overview

This system utilizes a **Time-Domain Synthetic Speech Detection (TSSD)** architecture combined with a **multi-head attention mechanism** to capture both local and global audio features. The model is trained on a dataset of 30,000 audio samples (15k real + 15k fake), and deployed using a real-time Flask web interface for practical use.

## 🎯 Objective

To develop a high-accuracy, real-time audio classification system that detects deepfake speech using time-series neural networks and advanced attention-based learning.

## ✨ Features

- **TSSD Architecture**: Specialized 1D CNN model tailored for temporal feature extraction in audio.
- **Attention Mechanism**: Multi-head self-attention layer enhances focus on critical segments of speech.
- **Rich Feature Extraction**: Uses MFCC, Mel Spectrogram, Chroma, ZCR, Spectral Centroid & Flatness.
- **High Performance**:  
  - ✅ Accuracy: **98.20%**  
  - ✅ EER: **0.0505**  
  - ✅ T-DCF: **0.0044**
- **Web Interface**: Real-time prediction with audio upload and user login through Flask.

## 🛠️ Tech Stack

- **Python** – Core programming language
- **TensorFlow / Keras** – Model training and evaluation
- **Librosa** – Audio preprocessing and feature extraction
- **Flask** – Web framework for UI and backend integration
- **HTML/CSS** – Frontend for file upload and result display

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



