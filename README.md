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





# 🔐 Transformer-Based Speech Anonymization

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Audio-librosa-blue)](https://librosa.org/)
[![Model](https://img.shields.io/badge/Model-Transformer-informational)](https://arxiv.org/abs/1706.03762)

A TensorFlow implementation of a **Transformer-based model for speech anonymization**, which transforms Mel-spectrograms to anonymized spectrograms while preserving content. Ideal for privacy-preserving audio processing in voice assistants, healthcare, or public dataset anonymization.

---

## 📌 Features

- 🎧 Converts raw audio into Mel-spectrograms using `librosa`
- 🧠 Uses Positional Encoding and Multi-Head Attention layers
- 🔁 Residual connections and Layer Normalization for deep modeling
- 🛡️ Custom loss to balance anonymization vs. content preservation
- 🧪 Ready for experimentation with dummy or real datasets

---

## 📁 Project Structure

- `preprocess_audio`: Converts `.wav` to Mel-spectrogram
- `PositionalEncoding`: Adds temporal context
- `build_transformer_model`: Constructs the full model
- `loss_fn`: Custom loss for privacy + accuracy
- `train`: Example training setup with dummy data

---

## 🚀 Quick Start

1. **Install dependencies**

```bash
pip install tensorflow librosa numpy
```

2. **Train with Dummy Data**

```python
# From Python
transformer_model.fit(dummy_data, dummy_labels, epochs=10)
```

3. **Inference**

```python
anonymized = transformer_model.predict(mel_spectrogram_input)
```

---

## 📊 Model Architecture

- Input: Mel-spectrogram [128, 128]
- Layers:
  - Positional Encoding
  - 4 × [Multi-Head Attention + Feedforward + LayerNorm + Residual]
  - Final Dense Layer
- Output: Anonymized Mel-spectrogram

---

## 🤖 Use Cases

- Privacy-preserving speech analytics
- De-identifying clinical recordings
- Secure voice assistant training

---

## 📄 License

MIT © 2025

---

## ✨ Acknowledgements

Inspired by the original Transformer paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Built with ❤️ using TensorFlow and Librosa.

