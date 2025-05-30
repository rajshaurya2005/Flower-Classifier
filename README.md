# Iris Classification Neural Network

A PyTorch implementation of a neural network classifier for the famous Iris dataset. This project demonstrates basic machine learning concepts including data preprocessing, neural network architecture, and model evaluation.

## Overview

The Iris dataset contains 150 samples of iris flowers with 4 features (sepal length, sepal width, petal length, petal width) across 3 species (Setosa, Versicolor, Virginica). This neural network achieves high classification accuracy using a simple feedforward architecture.

## Features

- Data loading from scikit-learn's built-in Iris dataset
- Preprocessing pipeline with standardization
- Neural network with 3 fully connected layers
- Training with validation monitoring
- Complete evaluation on test set

## Model Architecture

- Input layer: 4 features
- Hidden layer 1: 16 neurons with ReLU activation
- Hidden layer 2: 8 neurons with ReLU activation  
- Output layer: 3 neurons (one per class)
- Dropout regularization (0.2) applied after each hidden layer

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python iris_classifier.py
```

The script will:
1. Load and split the Iris dataset (60% train, 20% validation, 20% test)
2. Preprocess features using StandardScaler
3. Train the neural network for 200 epochs
4. Display training progress every 50 epochs
5. Evaluate final performance on the test set

## Expected Results

The model typically achieves 95%+ accuracy on the test set. Training progress is displayed showing train loss, validation loss, and validation accuracy.

## Dataset Split

- Training: 90 samples (60%)
- Validation: 30 samples (20%) 
- Test: 30 samples (20%)

## Requirements

- Python 3.7+
- PyTorch 2.0+
- scikit-learn 1.3+
- NumPy 1.24+
