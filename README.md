# 🧠 Fashion-MNIST Classification using CNN (TensorFlow / Keras)

## 📄 Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images from the **Fashion-MNIST dataset** — a dataset of grayscale images representing various fashion items such as shirts, shoes, bags, and more.

The code covers the **entire deep learning workflow**, from dataset loading and preprocessing to training, evaluation, visualization, and saving the model.

---

## 👨‍💻 Author
- **Name:** [Your Name]  
- **Course:** AI / Deep Learning Assignment  
- **Institution:** [Your Institution Name]  
- **Date:** [Insert Date]

---

## 🧩 Features
- ✅ Reproducible training with fixed random seeds  
- ✅ Data augmentation for better generalization  
- ✅ CNN architecture with Batch Normalization & Dropout  
- ✅ Training callbacks: Model Checkpoint, Early Stopping, Learning Rate Reduction, CSV Logger  
- ✅ Visualization of training accuracy and loss curves  
- ✅ Automatic saving of model, best checkpoint, and training logs  
- ✅ Example predictions from the trained model  

---

## 📦 Requirements

Before running the code, install the required Python packages:

```bash
pip install tensorflow numpy pandas matplotlib
📁 Fashion-MNIST-CNN/
│
├── fashion_mnist_cnn.py          # Main Python script (this file)
├── checkpoints/                  # Saved best model (.h5)
├── history.csv                   # Training history (CSV)
├── training_log.csv              # Live training log from CSVLogger
├── training_curves_student.png   # Accuracy & loss plot
└── fashion_mnist_cnn_final.h5    # Final saved model
python fashion_mnist_cnn.py
Model: "fashion_mnist_cnn_final"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
...
Total params: ~350,000
Trainable params: ~349,000
Non-trainable params: ~1,000
_________________________________________________________________
