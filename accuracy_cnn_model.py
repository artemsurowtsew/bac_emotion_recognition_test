# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models  import model_from_json
import matplotlib.pyplot as plt
import seaborn as sns
# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)
import numpy as np
# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'C:/Users/Artem/Desktop/data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'C:/Users/Artem/Desktop/data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

json_file = open(r'C:\Users\Artem\PycharmProjects\pythonProject\emotion_model1505.json')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Завантаження ваг з H5-файлу
emotion_model.load_weights(r"C:\Users\Artem\PycharmProjects\pythonProject\emotion_model1505.h5")

print("Loaded model architecture and weights from disk")

emotion_model.summary()
