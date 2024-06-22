
# Emotion detection Project




## 1. Introduction
The goal of this project is to develop a machine learning application that can recognize and classify emotions (e.g., happy,
sad, angry, surprised) from images or real-time video streams.
## 2. Data Preparation (optional)
The original FER2013 dataset in Kaggle (https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing.

In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the dataset_prepare.py file which can be used for reference.


## 3. Key features
1. Dataset Preparation: Use a pre-existing dataset or collect and preprocess your own dataset of facial expressions.
2. Model Training: Train a machine learning model to classify facial expressions.
3. Face Detection: Use a face detection algorithm to detect faces in images or video streams.
4. Emotion Recognition: Use the trained model to recognize and classify emotions.
5. User Interface: Create a simple interface to display results.
## 4. Tools and technologies 
1. Programming Language: Python
2. Libraries and Frameworks:
OpenCV for face detection and image processing
TensorFlow or PyTorch for training the neural network
Keras (if using TensorFlow) for building and training models
scikit-learn for preprocessing and evaluating the model
Tkinter or Streamlit for creating a simple user interface