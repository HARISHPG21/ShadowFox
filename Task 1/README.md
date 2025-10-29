# 🧠 Image Classification with CNN – CIFAR-10  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR10-orange)](https://www.cs.toronto.edu/~kriz/cifar.html)  
[![License](https://img.shields.io/badge/License-MIT-green)](#)  
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)](#)  

---

## 🏁 Overview  

This folder contains a deep learning project that uses a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**.  
The model learns to recognize everyday objects like **cats**, **dogs**, **cars**, and **ships**, and is implemented using **PyTorch**.  

It demonstrates best practices in model training, validation, and evaluation — optimized for performance and deployment readiness.  

---

## 🎯 Objective  

Build a CNN-based classifier that can accurately identify small color images into one of ten object categories.  
The model focuses on achieving high generalization, efficient computation, and a clean, modular codebase.  

---

## 📦 Dataset  

- **Source:** [`torchvision.datasets.CIFAR10`](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)  
- **Classes:**  
  `['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`  
- **Samples:** 50,000 for training (split into train/validation) and 10,000 for testing  
- **Image Size:** 32×32 pixels (RGB)  

---

## ⚙️ Workflow  

1. **🔍 Device Detection**  
   Automatically detects and uses the available device (CUDA / MPS / CPU).  

2. **🧼 Data Preprocessing**  
   - Normalization of pixel values  
   - Conversion from PIL images to tensors  

3. **🎨 Data Augmentation**  
   Enhances generalization using:  
   - Random horizontal flips  
   - Random cropping  
   - Color jittering  

4. **🧠 Model Architecture**  
   - 3 convolutional layers with ReLU & MaxPooling  
   - Batch Normalization and Dropout for regularization  
   - Fully connected layers  
   - LogSoftmax output activation  

5. **⚡ Training Setup**  
   - Optimizer: **Adam**  
   - Loss Function: **NLLLoss**  
   - Validation monitoring at each epoch  

6. **📊 Evaluation Metrics**  
   - Accuracy, Precision, Recall, and F1-Score  
   - Confusion matrix visualization  

7. **💾 Model Saving**  
   - Trained model stored as `cnn_cifar10.pth`  

---

## 📈 Results  

- ✅ **Test Accuracy:** ~84%  
- 🧩 **Observations:** Strong balance between bias and variance with minimal overfitting  
- 📊 **Insights:** Classification report and confusion matrix show robust performance across classes  

---

## 🧰 Tech Stack  

- **Language:** Python  
- **Libraries:**  
  - PyTorch & Torchvision  
  - NumPy, Matplotlib, Seaborn  
  - Scikit-learn  

---

## 📁 Repository Structure  

| File | Description |
|------|--------------|
| `cnn_cifar10.pth` | Trained CNN model weights |
| `model.py` | CNN model definition |
| `train.py` | Training and validation pipeline |
| `evaluate.py` | Evaluation metrics and plots |
| `utils.py` | Helper functions for prediction and visualization |
| `README.md` | Project documentation |

---

## 🚀 Future Enhancements  

- Integrate **EarlyStopping** and **learning rate schedulers**  
- Experiment with **ResNet18** or **EfficientNet** for improved accuracy  
- Extend to **custom datasets** or **real-time webcam input**  
- Deploy using **Flask** or **Streamlit** as an interactive web app  

---

## 💡 Key Takeaway  

This project highlights how a compact CNN, trained with proper preprocessing and augmentation, can achieve solid results on CIFAR-10.  
It serves as both a learning resource and a baseline for more advanced computer vision experiments.  

---

### 🌟 Author  

Developed with ❤️ using **PyTorch** and **deep learning curiosity**.  
