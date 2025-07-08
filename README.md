# Facial Expression Detection & Recognition

This project uses a **Convolutional Neural Network (CNN)** built with Keras/TensorFlow to detect and recognize human facial expressions from grayscale images (48x48 pixels).

## 🔍 Overview

- **Input**: 48x48 grayscale facial images  
- **Output**: One of 7 emotions – *Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*  
- **Dataset**: [FER-2013 (Facial Expression Recognition 2013)](https://www.kaggle.com/datasets/msambare/fer2013)  
- **Model Summary**:
  - Stacked **Conv2D**, **BatchNorm**, **ReLU**, **Dropout**, and **MaxPooling** layers
  - Ends with fully connected (**Dense**) layers and **softmax** output
  - Total output classes: **7**

## 🧠 Architecture

The architecture includes:
- 5 Conv2D blocks with increasing filters (32 → 128)
- BatchNormalization & ReLU after each Conv layer
- Dropout layers for regularization
- Flatten → Dense (250 units) → Dropout → Dense (7 output classes)

> Visual representation is available in `best_model.png`.

## 📂 Files Included

- `Facial Expression Recognition.json` – Model architecture  
- `fer.weights.h5` – Pre-trained weights  
- `Facial_Expression_Detection_System.hdf5` – Saved model file  
- `best_model.png` – Architecture diagram

## 📊 Dataset

- **FER2013**: 35,887 labeled images of 48x48 grayscale faces.
- Source: [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
