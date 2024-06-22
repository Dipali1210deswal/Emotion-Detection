import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# Set the path to the model
model_path = 'C:/Users/DeLL/OneDrive/Desktop/dipali/minor proj/Emotion detection/emotion_model.keras'

# Load the model
model = load_model(model_path, compile=False)  # Setting compile=False as we don't need to compile for inference

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess image and make prediction
def predict_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, (48, 48))
    img_array = image.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    max_index = int(np.argmax(predictions))
    predicted_emotion = emotion_labels[max_index]
    return predicted_emotion

# Streamlit app interface
st.title('Emotion Detection App')
st.write('Upload an image and the app will predict the emotion.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR")

    if st.button('Predict Emotion'):
        emotion = predict_emotion(img)
        st.write(f'The predicted emotion is: {emotion}')
