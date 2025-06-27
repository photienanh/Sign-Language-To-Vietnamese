# VIETNAMESE SIGN LANGUAGE RECOGNITION
A system for recognizing Vietnamese Sign Language using deep learning and computer vision techniques, tailored specifically for Vietnamese sign language.
## Demo
https://github.com/user-attachments/assets/c143c7f2-9a7c-4033-9c41-a196322e6b5d
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Training from Scratch](#training-from-scratch)
## Overview
The Vietnamese Sign Language Recognition system leverages deep learning models and computer vision to interpret Vietnamese sign language gestures. It uses MediaPipe for landmark detection, TensorFlow for model training, and Streamlit for a user-friendly interface. The system supports recognition through video files or live webcam feeds.
## Features
- Automated Video Download: Automatically downloads videos for training data.
- Data Preprocessing: Processes and augments data for model training.
- Sign Language Recognition: Recognizes Vietnamese sign language gestures via video or webcam input.
- User Interface: Provides a Streamlit-based web interface for easy interaction.
## Requirements
- **Software**:
    - Python 3.8 or higher
    - TensorFlow 2.x
    - Scikit-learn
    - MediaPipe
    - OpenCV
    - Streamlit
- **Hardware**:
    - Webcam (required for webcam recognition)
    - GPU (recommended for model training)
## Installation
### 1. Clone the repository
```bash
git clone https://github.com/photienanh/Vietnamese-Sign-Language-Recognition
cd Vietnamese-Sign-Language-Recognition
```
Alternatively, download the ZIP file from GitHub and extract it.
### 2. Install Dependencies
Ensure Python is installed. If not, you can download and install it from the official [Python website](https://www.python.org/downloads/). Then, install the required libraries:
```bash
pip install -r requirements.txt
```
## Usage
The system can be used either by running the pre-trained model or by training a new model from scratch.
### Running the Application
To use the pre-trained model with the Streamlit interface:
```bash
streamlit run main.py
```
This launches a web interface where you can upload videos or use a webcam for sign language recognition.
### Training from Scratch
To train a new model, follow these steps:
1. Clear Previous Data (optional).
```bash
Get-ChildItem -Path "./" -Directory | Remove-Item -Recurse -Force
```
2. Download Training Data.
```bash
python download_data.py
```

3. Process Data.
```bash
python create_data_augment.py
```

4. Train the Model.
- Open ```training.ipynb``` in a Jupyter Notebook environment.
- Run all cells to train the model.
- Note: Training is computationally intensive and best performed on a GPU-enabled device.
5. Run the Application.
```bash
streamlit run main.py
```
