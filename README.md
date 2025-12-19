# Latin Letter Detection Project

This project aims to build and optimize a machine learning model to detect Latin letters, focusing on training a neural network for character recognition using the EMNIST dataset. The project also includes a model quantization process for deployment on embedded systems (STM).

## Project Structure

### File Breakdown

1. **artifacts/**
   This directory contains the saved models, quantization artifacts and other intermediate files used during training and evaluation. These files are crucial for the deployment of the model in real-world applications, especially for embedded systems.

2. **emnist/**
   This folder contains the dataset files related to the **EMNIST** (Extended MNIST) dataset, which is used to train and evaluate the character detection model. It consists of handwritten characters similar to the MNIST dataset, but includes letters as well as digits.

3. **handwrite_app.py**
   This script is a simple application for testing the trained model. It allows for drawing characters that are then processed by the model to predict the corresponding Latin letter. It serves as a user interface for practical use cases of the model.

4. **mnist.npz**
   This file contains the preprocessed data in **NumPy** format. It includes both images and labels used for training and evaluation. The data is formatted to be compatible with the training pipeline in **train_and_quantize.py**.

5. **requirements.txt**
   This file lists all the necessary Python libraries and dependencies needed to run the project. It includes libraries like TensorFlow, PyTorch, NumPy, etc. To install these dependencies, simply run:

   ```bash
   pip install -r requirements.txt
   ```

6. **STM/**
   This directory likely contains the necessary files and configurations for deploying the model to an **STM32** microcontroller or similar embedded system. It includes any scripts, configurations, or libraries needed for the quantized model to run on hardware.

7. **train_and_quantize.py**
   This script contains the core logic for training the model and performing quantization. The quantization step reduces the model size and computation requirements, making it suitable for deployment on embedded devices. It includes:

   * Model architecture definition.
   * Training logic using the EMNIST dataset.
   * Post-training quantization using techniques like **post-training quantization** for optimization on edge devices.

## Latin Letter Detection

This project focuses on the detection of Latin letters using deep learning techniques. We utilize the EMNIST dataset, which is an extended version of the MNIST dataset, containing handwritten characters and digits. The goal of the project is to accurately detect the Latin alphabet from a variety of handwritten inputs, making it applicable for use cases such as:

* Optical Character Recognition (OCR).
* Handwriting-to-text applications.
* Embedded systems for real-time detection in mobile or hardware applications.

The model is trained on the EMNIST dataset and utilizes a Convolutional Neural Network (CNN) architecture to learn the spatial features of handwritten characters. After training, the model is further optimized using quantization to reduce its size and computation load, making it deployable on embedded devices.

## EMNIST Dataset

The **EMNIST** dataset is an extension of the original MNIST dataset, but it focuses on characters instead of just digits. EMNIST contains multiple variants, including:

* **ByClass**: Includes 814,255 characters from 814 classes (including 62 classes representing the uppercase and lowercase Latin alphabet).
* **ByMerge**: Contains 814 classes, merging some of the categories from ByClass.
* **Letters**: This specific variant contains 145,000 images of handwritten letters.

For this project, the **EMNIST Letters** dataset was used, which contains images of 26 uppercase and 26 lowercase Latin letters. These letters are handwritten by various individuals, providing a rich source of data for training a robust model for letter detection.

Link for NIST official website: https://www.nist.gov/itl/products-and-services/emnist-dataset

## Quantization Process (STM)

Quantization is the process of reducing the precision of the numbers used in a model's weights and activations, which results in a smaller model size and faster inference time. This is particularly useful for deploying deep learning models on embedded systems with limited computational resources.

In this project, we have used **post-training quantization**:

1. **Model Training**: A standard CNN model is trained on the EMNIST dataset to classify letters.
2. **Quantization**: After training, the model is quantized using a technique like **dynamic range quantization**, which reduces the model size from floating-point precision to integer precision, optimizing it for STM-based systems.
3. **Deployment**: The quantized model is ready for deployment on embedded devices (STM32), where the smaller model size and faster inference enable real-time character recognition even on limited hardware.

## How to Run

### 1. Install Dependencies

First, clone the repository and install all the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Train the Model

To train the model from scratch, run the following command:

```bash
python train_and_quantize.py
```

This will start the training process and the model will be saved in the **artifacts/** directory.

### 3. Test the Model (Handwriting App)

To test the model on drawn characters, run:

```bash
python handwrite_app.py
```

This will launch an application where you can draw a letter and the model will predict it.

### 4. Quantization for Embedded Systems (STM)

Once the model is trained, you can quantize it for STM-based systems. This is done automatically in the **train_and_quantize.py** script, which handles both training and quantization processes.

### 5. Deployment to STM

Once the quantized model is ready, transfer the files from the **STM/** directory to the embedded device, following the device's specific instructions for deployment.

## Conclusion

This project demonstrates how to train, optimize and deploy a Latin letter detection model using the EMNIST dataset. With quantization, the model is optimized for efficient deployment on embedded systems like STM32. This solution can be applied in OCR applications, handwriting recognition systems and other real-time embedded AI projects.
