# MNIST Handwritten Digit Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using Keras. The model achieves high accuracy by leveraging convolutional layers, max pooling, and dropout regularization.

## Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. It is divided into:
- 60,000 training images
- 10,000 testing images

## Model Architecture
The implemented CNN model has the following layers:
1. **Conv2D (32 filters, 3x3 kernel, ReLU activation)** - Extracts features from the input image.
2. **Conv2D (64 filters, 3x3 kernel, ReLU activation)** - Further feature extraction.
3. **MaxPooling2D (2x2 pool size)** - Reduces spatial dimensions to prevent overfitting.
4. **Dropout (25%)** - Regularization to avoid overfitting.
5. **Flatten** - Converts feature maps into a 1D array.
6. **Dense (128 neurons, ReLU activation)** - Fully connected layer for learning representations.
7. **Dropout (50%)** - Additional regularization.
8. **Dense (10 neurons, Softmax activation)** - Output layer for classification.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install keras tensorflow numpy
```

## Training
The model is trained using:
- **Loss function**: Categorical Crossentropy
- **Optimizer**: Adadelta
- **Metrics**: Accuracy
- **Batch Size**: 128
- **Epochs**: 15

To train the model, run:
```bash
python mnist_cnn.py
```

## Evaluation
After training, the model is evaluated on the test dataset, and the final accuracy is displayed.

## Results
The model achieves high accuracy on the MNIST dataset, demonstrating the effectiveness of CNNs for digit classification.

## Credits
This implementation is inspired by the official Keras example: [MNIST CNN](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py).
