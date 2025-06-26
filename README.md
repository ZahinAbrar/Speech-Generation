# GAN for Raw Speech Generation

This project implements a Generative Adversarial Network (GAN) in TensorFlow for generating raw audio waveforms trained on the LibriSpeech dataset.

## 📦 Features

- **Generator and Discriminator**: Basic MLP-based GAN architecture for speech generation.
- **Preprocessing**: Loads and normalizes raw `.flac` speech audio to fixed-length waveforms.
- **Training**: Full training loop with discriminator and generator losses, real/fake labeling, and periodic generation of output samples.
- **Visualization**: Plots generated audio waveform.
- **Export**: Saves generated waveform as a `.wav` file.

## 🗃 Dataset

The model uses the [LibriSpeech Train-Clean-100 subset](http://www.openslr.org/resources/12/train-clean-100.tar.gz). It's automatically downloaded and extracted.

## 🧪 Usage

### Installation
```bash
pip install tensorflow librosa soundfile wget
```

### Download and Extract Dataset
```bash
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvzf train-clean-100.tar.gz
```

### Training
```python
train_gan(generator, discriminator, gan, epochs=5000, batch_size=32, audio_files=audio_files)
```

### Saving Generated Samples
```python
save_audio_sample(generator, noise_dim=100, epoch=100)
```

### Download Output (Google Colab)
```python
from google.colab import files
files.download('generated_speech_100.wav')
```

## 🧠 Model Summary

- Generator: 3 fully connected layers (Dense) with `tanh` output activation.
- Discriminator: Binary classifier with sigmoid activation to distinguish real vs. generated waveforms.
- Input Dimension: 100-D noise vector.
- Output: 1D raw audio signal (16k samples ~1 second at 16kHz).

## 🎯 Applications

- Research in low-resource speech generation
- Proof-of-concept for GAN-based speech synthesis
- Educational project for audio ML

## 🚧 Notes

- Ensure enough disk space for LibriSpeech (approx. 6GB for full subset).
- Output quality improves over training epochs, but model is intentionally minimal.

## 📁 Output Example

Generated speech samples are saved as `generated_speech_<epoch>.wav` and plotted using `matplotlib`.



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
