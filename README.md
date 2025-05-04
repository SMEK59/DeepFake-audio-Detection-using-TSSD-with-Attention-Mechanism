# DeepFake-audio-Detection-using-TSSD-with-Attention-Mechanism

A deep learning-based system for detecting synthetic (deepfake) audio using a Time-Domain Synthetic Speech Detection (TSSD) model enhanced with a multi-head attention mechanism. The system analyzes key spectral and temporal features such as MFCC, Mel Spectrogram, and Chroma to classify audio clips as real or fake with high accuracy. It includes a real-time Flask web interface for audio upload, user authentication, and instant detection results.

## 🚀 Features

- Real vs. fake audio classification using deep learning
- Audio feature extraction using signal processing techniques
- TSSD model with residual CNN blocks and attention mechanism
- Real-time prediction via Flask-based web app
- User login and audio file upload functionality

## 🧠 Model Architecture

- **1D Convolutional Neural Networks (CNNs)**
- **Residual Blocks (RSM1D)** for enhanced gradient flow
- **Multi-Head Attention Mechanism** for focusing on key audio regions
- **Dense Layers with Softmax** for binary classification

## 📊 Performance Metrics

| Model                | Accuracy | EER    | T-DCF   |
|---------------------|----------|--------|---------|
| TSSD + Attention    | 98.2%    | 0.0505 | 0.0044  |
| TSSD                | 94.5%    | 0.0905 | 0.0244  |
| Multilayer Perceptron | 88.0%  |   -    |   -     |

- **AUC Score**: 0.99  
- Metrics indicate significant improvement over baseline models

## 📁 Dataset

- **Total Samples**: 30,000 audio clips  
  - 15,000 real (from open datasets of human speech)
  - 15,000 fake (generated using WaveNet and other TTS models)

## 🛠️ Technologies Used

- **Languages**: Python  
- **Libraries**: TensorFlow, Keras, Librosa, NumPy, Matplotlib  
- **Web Framework**: Flask  
- **Frontend**: HTML, CSS

## 📦 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/deepfake-audio-detection.git
cd deepfake-audio-detection

# Install required packages
pip install -r requirements.txt

# Run the Flask web app
python app.py


