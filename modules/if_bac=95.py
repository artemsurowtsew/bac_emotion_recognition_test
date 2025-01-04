from PyQt5 import QtCore, QtGui, QtWidgets
import time
import cv2
import mss
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

json_file = open(r'C:\Users\Artem\PycharmProjects\pythonProject\model\emotion_model3.json')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Завантаження ваг з H5-файлу
emotion_model.load_weights(r"C:\Users\Artem\PycharmProjects\pythonProject\model\emotion_model3.h5")

print("Loaded model architecture and weights from disk")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier( r'C:\Users\Artem\PycharmProjects\pythonProject\haarcascades/haarcascade_frontalface_default.xml')

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Настройка интерфейса
        self.setWindowTitle("EmotionRecognition")
        self.setGeometry(300, 300, 800, 600)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(10, 10, 1280, 720))

        # Установка таймера для захвата экрана
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.capture_screen)
        self.timer.start(24)  # Захват кожні 24 мс

    def capture_screen(self):
        # Захват экрана
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            img = np.array(sct.grab(monitor))

        # Обработка изображения и распознавание эмоций
        self.process_frame(img)

        # Update the QLabel with the processed image
        self.update_image(img)

    def update_image(self, img):
        # Convert the image format to RGB for QPixmap
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def process_frame(self, frame):
        # Преобразование изображения в оттенки серого для обнаружения лиц
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Обработка каждого обнаруженного лица
        for (x, y, w, h) in faces:
            face_gray = gray_frame[y:y + h, x:x + w]  # Используем оттенки серого для обработки
            face_color = frame[y:y + h, x:x + w]  # Используем цветное изображение для отображения

            # Обработка face_gray с помощью вашей модели
            emotion_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
            face_gray_resized = cv2.resize(face_gray, (48, 48))
            face_gray_normalized = face_gray_resized / 255.0
            face_gray_reshaped = np.reshape(face_gray_normalized, (1, 48, 48, 1))
            predicted_emotion = emotion_model.predict(face_gray_reshaped)
            predicted_emotion_label = emotion_labels[np.argmax(predicted_emotion)]

            # Отображение результатов на face_color
            cv2.putText(frame, predicted_emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Update the QLabel with the processed image
        self.update_image(frame)


class Ui_EmotionRecognition(object):
    def setupUi(self, EmotionRecognition):
        EmotionRecognition.setObjectName("EmotionRecognition")
        EmotionRecognition.resize(767, 654)
        EmotionRecognition.setStyleSheet("QWidget {\n"
"    background-color: #808080; /* Сірий колір фону */\n"
"    color: #FFA500; /* Оранжевий колір тексту */\n"
"}\n"
"\n"
"QPushButton {\n"
"    background-color: #FFA500; /* Оранжевий колір фону кнопки */\n"
"    color: #808080; /* Сірий колір тексту кнопки */\n"
"}\n"
"\n"
"QLineEdit {\n"
"    background-color: #D3D3D3; /* Світло-сірий колір фону поля введення */\n"
"    color: #FFA500; /* Оранжевий колір тексту поля введення */\n"
"}")
        self.centralwidget = QtWidgets.QWidget(Emotion