from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QIcon, QFont
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(200, 200, 700, 400)
        self.setWindowTitle("Python GUI Development")
        self.setWindowIcon(QIcon('/Users/Artem/Desktop/logo.png'))
        label = QLabel("Hello World",self)
        label.setText("This is")
        label.move(100,100)
        label.setFont(QFont("Arial", 15));
        label.setStyleSheet('color:red')

        label.setText(str(12))
        label.setNum(15)