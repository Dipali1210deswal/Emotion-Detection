import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Data augmentation
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen_val = ImageDataGenerator(rescale=1./255)

# Load data
train_data = datagen_train.flow_from_directory('data/train', target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical')
val_data = datagen_val.flow_from_directory('data/test', target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical')

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Specify full path for saving the model in native Keras format (.keras)
model_save_path = 'C:/Users/DeLL/OneDrive/Desktop/dipali/minor proj/Emotion detection/emotion_model.keras'

# Save the model
save_model(model, model_save_path)

print("Model saved successfully at:", model_save_path)
