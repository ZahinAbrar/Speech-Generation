# 🎙️ GAN-Based Speech Generation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green)
![GAN](https://img.shields.io/badge/Model-GAN-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🎯 Overview

This project implements a Generative Adversarial Network (GAN) to synthesize human-like speech waveforms. It includes:

- ✅ A **Generator** model that produces 1-second speech samples.
- ✅ A **Discriminator** that classifies real vs. generated audio.
- ✅ **Preprocessing pipeline** for raw `.flac` audio files.
- ✅ Training on the **LibriSpeech** dataset.

---

## 🔧 Setup Instructions

```bash
pip install tensorflow librosa soundfile wget
```

Download and extract LibriSpeech:

```bash
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvzf train-clean-100.tar.gz
```

---

## 🧠 Architecture

### Generator
- Dense layers with ReLU activations
- Output layer with `tanh` to generate audio in [-1, 1]

### Discriminator
- Fully connected binary classifier

### GAN
- Combines Generator and Discriminator
- Trained using adversarial loss

---

## 🌀 Training

```python
train_gan(generator, discriminator, gan, epochs=5000, batch_size=32, audio_files=audio_files)
```

At every 100 epochs, a `.wav` file is generated and plotted.

---

## 📈 Sample Output

![Generated Audio](generated_plot.png)

*Above: waveform of synthesized speech at epoch 100*

---

## 🔊 Listen to Generated Audio

After training:
```python
files.download('generated_speech_100.wav')
```

---

## 📁 Project Structure

```
├── generator.py
├── discriminator.py
├── train.py
├── data/
│   └── LibriSpeech/
├── utils.py
└── README.md
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Acknowledgments

- LibriSpeech Dataset
- TensorFlow + Keras
- GAN inspiration from DeepLearning.ai




# 🎤 Transformer-based Speech Anonymization

This project implements a **Transformer-based neural architecture** for **speech anonymization**, where the model transforms an input Mel-Spectrogram into a modified version that conceals speaker identity while preserving linguistic content.

> ⚡ Goal: Protect speaker privacy while retaining the semantic integrity of speech.

---

## 🚀 Key Features

- ✅ **Mel-Spectrogram Preprocessing** with `librosa`
- ✅ **Custom Positional Encoding Layer** for time-frequency data
- ✅ **Multi-Head Attention Blocks** inspired by Transformer architecture
- ✅ **Custom Loss Function** balancing content preservation and anonymization
- ✅ **Modular and Scalable Design** for easy experimentation and extension

---

## 📁 Project Structure

```bash
.
├── transformer_anonymizer.py     # Main model and training script
├── utils.py                      # Preprocessing and utility functions
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🧠 Model Architecture

This project adapts the Transformer encoder design for 2D Mel-Spectrogram inputs:

- **Input:** `(T, M)` Mel-Spectrogram (Time × Mel-bins)
- **Layers:**
  - Positional Encoding
  - Stack of N Transformer encoder blocks:
    - Layer Normalization
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Residual Connections
- **Output:** Transformed Mel-Spectrogram with anonymized speaker characteristics

---

## 🧪 Custom Loss Function

```python
total_loss = α * ContentLoss + (1 - α) * AnonymizationLoss
```

- **Content Loss**: Encourages preserving the original content
- **Anonymization Loss**: Encourages dissimilarity in speaker identity features

This design supports tunable privacy-preservation tradeoffs via the `α` hyperparameter.

---

## 📊 Example Usage

```python
# Load and convert audio to Mel-spectrogram
mel = preprocess_audio("sample.wav")

# Predict anonymized version
output = transformer_model.predict(mel[np.newaxis, ...])
```

---

## 🛠️ Training

To train on dummy data (replace with real samples):

```python
transformer_model.fit(dummy_data, dummy_labels, batch_size=16, epochs=10)
```

---

## 🧰 Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

**Core libraries:**
- TensorFlow
- librosa
- NumPy

---

## 🎯 Applications

- Privacy-preserving voice assistants
- Anonymous call centers
- Speech data sharing with built-in de-identification
- Preprocessing for training fairer speech recognition models

---

## 📌 Next Steps

- Integrate speaker verification metrics to quantify anonymization
- Add evaluation with WER (Word Error Rate) to measure content preservation
- Support real-time inference using TensorFlow Lite or ONNX

---
