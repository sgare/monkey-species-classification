# Monkey Species Classification 🐒

A deep learning project for classifying monkey images into their respective species using Convolutional Neural Networks (CNNs). The repository includes data preprocessing, model training, evaluation, and visualization of results.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
This project demonstrates an end-to-end pipeline for monkey species classification:
- Preprocessing and augmentation of image data
- Building and training a CNN model
- Evaluating model performance using accuracy and confusion matrix
- Saving and exporting the trained model for inference

The model can classify multiple monkey species and is suitable for wildlife research, educational purposes, and machine learning practice.

---

## Dataset
The dataset consists of labeled monkey images belonging to different species. You can replace it with your own dataset in the same structure:  
dataset/
├── species_1/
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
├── species_2/
│ └── ...
└── species_n/
└── ...

---

## Features
- Image preprocessing and augmentation
- CNN-based model architecture
- Model training and evaluation
- Accuracy and confusion matrix visualization
- Save trained model for inference

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/monkey-species-classification.git
cd monkey-species-classification
pip install -r requirements.txt



## Usage
Prepare your dataset as described above.
Run the training script:
python train_monkey_classifier.py
Check accuracy_plot.png and confusion_matrix.png for model evaluation.
Use the saved model for inference on new images.

## Results
After training, the model achieves high accuracy in classifying monkey species. Visualizations include:
accuracy_plot.png: Model accuracy and loss over epochs
confusion_matrix.png: Model predictions vs actual labels
