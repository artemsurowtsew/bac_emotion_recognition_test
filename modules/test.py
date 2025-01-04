from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage
import cv2
import time
import mss
import mss.tools
import numpy as np


face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Artem\PycharmProjects\pythonProject\haarcascades/haarcascade_frontalface_default.xml')


class Worker_Screen(QThread):
    ImageUpdate2 = pyqtSignal(QImage)

    def __init__(self, ocv_conf, welcome):
        super().__init__()
        self.config = ocv_conf
        self.welcome = welcome

    def run(self):
        """Activate the video processing and charge frames to predict"""
        with mss.mss() as sct:
            monitor_num = 2
            mon = sct.monitors[monitor_num]
            monitor = {
                "top": mon["top"],
                "left": mon["left"],
                "width": mon["width"],
                "height": mon["height"],
                "mon": monitor_num,
            }
            self.ThreadActive = True
            i = 0
            x, y, w, h = 0, 0, 0, 0
            self.faces_gray = []
            while self.ThreadActive:
                img = sct.grab(monitor)  # tomamos un pantallazo
                frame = np.array(img)
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(Image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if i == 30:
                    if len(faces) > 0:
                        self.faces_gray = []
                        for (x, y, w, h) in faces:
                            careto = {}
                            careto["gray"] = gray[y:y+h, x:x+w]
                            careto["pos"] = (x, y, w, h)
                            careto["pred"] = get_prediction(gray[y:y+h, x:x+w])
                            self.faces_gray.append(careto)
                            self.welcome.emotions_screen_reg.append(
                                careto["pred"] + "  at  " + time.strftime("%X"))
                            print(careto["pred"])
                            print(
                                self.welcome.counters_screen[careto["pred"]]['val'])
                            self.welcome.counters_screen[careto["pred"]
                                                         ]['val'] += 1
                            self.welcome.counters_screen[careto["pred"]]['lcd'].display(
                                self.welcome.counters_screen[careto["pred"]]['val'])

                        i = 0
                else:
                    i += 1
                for cara in self.faces_gray:
                    x, y, w, h = cara["pos"]
                    cv2.putText(Image,
                                cara["pred"],
                                (x, y),
                                self.config['font'],
                                self.config['fontScale'],
                                self.config['fontColor'],
                                self.config['lineType'])

                ConvertToQtFormat = QImage(
                    Image.data, Image.shape[1], Image.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(
                    640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate2.emit(Pic)
            cv2.destroyAllWindows()

    def stop(self):
        """Desactivate the thread stopping the video capture"""
        self.ThreadActive = False
        self.quit()