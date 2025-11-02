# 🧠 Digit Recognizer — NumPy Deep Learning Implementation  

This project implements a complete **Neural Network (NN)** and **Convolutional Neural Network (CNN)** from scratch using only **NumPy** — no TensorFlow, no PyTorch, no Keras.  
Every component — from **forward propagation**, **backpropagation**, and **loss derivatives** — is built manually to understand how deep learning works under the hood.

---

## 🚀 Project Highlights

✅ **Manual Backpropagation Algorithm**  
- Implemented gradient computation for all layers without relying on autograd.  
- Includes step-by-step propagation of errors through the network weights and activations.

✅ **Custom Forward Propagation Engine**  
- Fully vectorized forward pass using NumPy for efficient matrix operations.  
- Supports both dense and convolutional layers.

✅ **Hand-Derived Loss Derivatives**  
- Implemented derivative functions for loss functions such as **Mean Squared Error (MSE)** and **Cross-Entropy Loss**.  
- Provides clear mathematical intuition on how loss gradients influence parameter updates.

✅ **Neural Network (NN) & Convolutional Neural Network (CNN)**  
- Implemented both feedforward and convolutional architectures.  
- CNN version enhances feature extraction and achieves higher accuracy on MNIST digits.

---

## 🧩 Architecture Overview

### 1. Neural Network (NN)
- **Input → Hidden Layer(s) → Output**  
- Activation: *ReLU / Sigmoid / Softmax*  
- Loss: *Cross-Entropy*  

### 2. Convolutional Neural Network (CNN)
- **Conv Layer → Pooling → Flatten → Dense → Output**  
- Filter weights and pooling implemented from scratch  
- Backpropagation through convolution and pooling layers

---

## 🧮 Mathematical Foundation

Each component is implemented from first principles:

- **Forward propagation**  
  \[
  z = W \cdot x + b
  \]
  \[
  a = f(z)
  \]

- **Backward propagation**  
  \[
  \frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial W}
  \]

- **Weight update rule**  
  \[
  W := W - \eta \cdot \frac{\partial L}{\partial W}
  \]

This ensures complete transparency and control over each training step.

---

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset (28×28 grayscale images)**, consisting of:

- 60,000 training images  
- 10,000 test images  

All preprocessing — normalization, reshaping, and one-hot encoding — is handled in pure NumPy.

---

## ⚙️ Installation & Usage

```bash
git clone https://github.com/yourusername/mnist-from-scratch.git
cd digit_recognizer_from_scratch
mnist_from_scratch.ipynb

