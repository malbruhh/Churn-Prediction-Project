# Customer Churn Prediction: Neural Network from Scratch

This repository contains a **custom implementation of a Multi-Layer Perceptron (MLP)** built **from scratch using NumPy**. The project focuses on predicting **customer churn** using a dataset retrieved from **UCI Machine Learning Repository**, featuring a complete **data preprocessing pipeline**.

---

## 🚀 Overview

The primary goal of this project is to **demonstrate the internal mechanics of Backpropagation and Gradient Descent** without relying on high-level deep learning frameworks such as **TensorFlow** or **PyTorch**.

---

## 📂 Dataset

**Source:** UCI Machine Learning Repository  
Iranian Churn Dataset  
https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset

### Feature Description

- **Call Failures:** Number of call failures  
- **Complains:** Complaint status (0 = No, 1 = Yes)  
- **Subscription Length:** Total subscription duration (months)  
- **Charge Amount:** Ordinal charge level (0–9)  
- **Seconds of Use:** Total call duration in seconds  
- **Frequency of Use:** Total number of calls made  
- **Frequency of SMS:** Total number of SMS sent  
- **Distinct Called Numbers:** Number of unique called phone numbers  
- **Age Group:** Ordinal age category (1–5)  
- **Tariff Plan:** Subscription type (1 = Pay-as-you-go, 2 = Contractual)  
- **Status:** Customer status (1 = Active, 2 = Non-active)  
- **Customer Value:** Calculated customer value score  
- **Churn:** Target label (1 = Churn, 0 = Non-churn)

---

### Key Features

- **Custom Neural Network Class**
  - Modular implementation
  - Configurable hidden layers
  - Custom activation functions
  - Adjustable learning rates

- **Data Preprocessing Pipeline**
  - **Log Transformation** to handle skewed numerical data and outliers  
  - **One-Hot Encoding** for categorical features:
    - Complains
    - Tariff Plan
    - Status
  - **Min-Max Scaling** to normalize feature ranges

- **Performance Evaluation**
  - Loss curves
  - Accuracy curves
  - Confusion matrices

---

## 🧪 Model Experiments

  ### 1️⃣ 1st Model: Onw Hidden Layer

- **Network Architecture:** 1 hidden layer  
- **Hidden Activation:** ReLU  
- **Output Activation:** Sigmoid  
- **Learning Rate:** 0.01  
- **Epochs:** 500  

---

### 2️⃣ 2nd Model: Two Hidden Layers

- **Network Architecture:** 2 hidden layers  
- **Hidden Activation:** ReLU  
- **Output Activation:** Sigmoid  
- **Learning Rate:** 0.01  
- **Epochs:** 500  

---

### 3️⃣ 3rd Model: Adaptive Learning Rate

- **Network Architecture:** 2 hidden layers  
- **Hidden Activation:** ReLU  
- **Output Activation:** Sigmoid  
- **Optimization:** Adaptive learning rate (Exponential decay)  
- **Epochs:** 500  

---

### 4️⃣ 4th Model: Larger Hidden Layers

- **Network Architecture:** 2 hidden layers (larger node count)  
- **Hidden Activation:** ReLU  
- **Output Activation:** Sigmoid  
- **Learning Rate:** 0.01  
- **Epochs:** 500  

---

### 5️⃣ 5th Model: No Mini-Batch (Full Batch)

- **Network Architecture:** 2 hidden layers  
- **Hidden Activation:** ReLU  
- **Output Activation:** Sigmoid  
- **Optimization:** Full-batch Gradient Descent  
- **Learning Rate:** 0.01  
- **Epochs:** 500  

---

### 6️⃣ 6th Model: Softmax Activation Function

- **Network Architecture:** 2 hidden layers  
- **Hidden Activation:** ReLU  
- **Output Activation:** Softmax  
- **Learning Rate:** 0.01  
- **Epochs:** 500  

---

## 🛠️ Installation & Requirements

To run the Jupyter Notebooks (`.ipynb` files), install the required libraries (included inside the `.ipynb` file):

```bash
pip install jupyter numpy pandas matplotlib seaborn tabulate imblearn
