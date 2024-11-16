# Project: Pneumonia Classification with Deep Learning

## Overview

This project aims to classify X-ray images as either "Normal" or "Pneumonia" using deep learning techniques, particularly convolutional neural networks (CNN) and transfer learning with VGG19.

## Features
- Data augmentation to improve generalization
- Transfer learning using the VGG19 architecture
- Flask-based web interface for model inference


## Setup Instructions

To get started with the project, follow these steps:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-username/pneumonia-classification.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application:
    ```bash
    python app.py
    ```

4. Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the web interface.

## Model Details

The model uses a pre-trained VGG19 model as the base and is fine-tuned for the task of pneumonia detection in X-ray images. The final output layer has 2 units corresponding to the two classes: **Normal** and **Pneumonia**.




