# Deep Neural Networks for Nonlinear Feature Extraction

## Description

This project involves designing and training **fully connected deep neural networks** on the **MNIST dataset** to extract bottleneck features for classification tasks. The model is fine-tuned to optimize performance for digit classification, achieving the best configuration for specific digit pairs using **SVM classifiers**.

## Features

- **Fully connected deep neural networks:** Designed for feature extraction and classification.
- **Feature extraction from MNIST dataset:** Capturing key representations from images.
- **SVM-based classification tuning:** Optimizing classification performance for specific digit pairs.

## Technologies Used

- **Python**
- **JAX**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **MNIST Dataset**

## Installation

```bash
pip install jax jaxlib numpy scikit-learn matplotlib mnist
```

## Dataset

The **MNIST dataset** is used, which contains handwritten digit images (0-9). The deep neural network extracts features, and **SVM classifiers** are used for classification tasks.

## Project Tasks

1. **Load and preprocess MNIST data using the mnist library.**
2. **Implement deep neural networks using JAX for feature extraction.**
3. **Extract bottleneck features for classification.**
4. **Optimize network architecture for better performance.**
5. **Apply SVM classifiers from scikit-learn to fine-tuned features.**
6. **Evaluate model accuracy and efficiency.**

## Visualizations

- **Training Loss and Accuracy:** Track model learning progress.
- **SVM Classification Results:** Evaluate classification performance.

