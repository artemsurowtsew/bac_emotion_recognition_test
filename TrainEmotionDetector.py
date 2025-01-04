
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

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

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# Збереження архітектури моделі у форматі JSON
model_json = emotion_model.to_json()
with open("emotion_model1905.json", "w") as json_file:
    json_file.write(model_json)

# Збереження навчених ваг моделі у файлі .h5
emotion_model.save_weights('emotion_model1905.h5')

# Відображення історії точності
plt.plot(emotion_model_info.history['accuracy'], label='Точність')
plt.plot(emotion_model_info.history['val_accuracy'], label='Точність на валідації')
plt.title('Історія точності моделі')
plt.ylabel('Точність')
plt.xlabel('Кількість циклів')
plt.legend()
plt.show()

import json

# Після тренування моделі збережіть історію у файл JSON
with open('emotion_model_history.json', 'w') as f:
    json.dump(emotion_model_info.history, f)

# Потім, ви можете завантажити історію з файлу
with open('emotion_model_history.json', 'r') as f:
    history = json.load(f)

