import os

from keras.src.saving.saving_api import load_model, load_weights
from keras.models import model_from_json

data_dir = r'C:\Users\Artem\Desktop\data'
os.listdir(data_dir)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.image import imread
test_path = data_dir+'\\test\\'
train_path= data_dir+'\\train\\'
test_path
os.chdir(test_path)

json_file = open(r'C:\Users\Artem\PycharmProjects\pythonProject\model\fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Завантаження ваг з H5-файлу
emotion_model.load_weights(r"C:\Users\Artem\PycharmProjects\pythonProject\model\fer.h5")

from keras.preprocessing import image

# Завантажте зображення з диску
img_path = r'C:\Users\Artem\Desktop\data\test\sad\PrivateTest_2013992.jpg'  # Замініть це на шлях до вашого зображення
img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')

# Перетворіть зображення в масив numpy
img_array = image.img_to_array(img)

# Додайте додатковий розмір для пакету
img_array = np.expand_dims(img_array, axis=0)

# Зробіть прогноз на зображенні
predictions = emotion_model.predict(img_array)

# Виведіть прогнозований клас
predicted_class = np.argmax(predictions[0])
print('Predicted class:', predicted_class)