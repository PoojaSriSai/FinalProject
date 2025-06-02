# 🧠 Diabetic Retinopathy Classification using ConvNeXt-B
This project implements a deep learning pipeline for multi-class classification of diabetic retinopathy using the ConvNeXt-Base model with CLAHE preprocessing, Albumentations-based augmentations, and Focal Loss for handling class imbalance. The IDRiD dataset is used for training and validation.
## 📌 Project Overview

- **Goal**: Classify retinal images into 5 DR grades (0–4).
- **Model**: ConvNeXt-Base pretrained on ImageNet.
- **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Augmentation**: Rotation, flip, brightness/contrast, normalization.
- **Loss Function**: Focal Loss for class imbalance.
- **Evaluation**: Accuracy and loss plots for training and validation.
## 📁 Dataset
Source: IDRiD Dataset
## 🧪 Environment & Libraries
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Albumentations
- Pandas, NumPy, Matplotlib
- scikit-learn
## 🧼 Preprocessing
CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied on the L channel of LAB images to enhance contrast and visibility.
Images are resized to 384×384.
## 📦 Model Architecture
Base model: ConvNeXtBase from tensorflow.keras.applications
Input shape: 384x384x3
Custom top layers:
Global Average Pooling
Dropout (0.5)
Dense softmax output layer (5 classes)
## 🔁 Training Configuration
- Batch size: 16
- Epochs: 50
- Optimizer: Adam (learning rate: 2e-4)
- ModelCheckpoint (best model saved as best_dr_model.keras)
## 📈 Future Work
Integrate Grad-CAM for visual explanations.

Test on external validation datasets.

Deploy as a Flask or Streamlit web app.
