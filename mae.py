from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models  import model_from_json
import matplotlib.pyplot as plt
import json


# Потім, ви можете завантажити історію з файлу
with open(r"C:\Users\Artem\PycharmProjects\pythonProject\emotion_model1505.json", 'r') as f:
    history = json.load(f)

    plt.plot(history.history['accuracy'], label='Точність')
    plt.plot(history.history['val_accuracy'], label='Точність на валідації')
    plt.title('Історія точності моделі')
    plt.ylabel('Точність')
    plt.xlabel('Кількість циклів')
    plt.legend()
    plt.show()