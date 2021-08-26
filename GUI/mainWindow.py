import os
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy
import torch.cuda
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import detect
from GUI import terminalWindow


class Ui_MainWindow(QMainWindow):
    progress = pyqtSignal(int)
    display_signal = pyqtSignal(str)
    stop_signal = pyqtSignal()

    # next_frame = pyqtSignal(numpy.ndarray)

    def __init__(self):
        super().__init__()
        self.height = QApplication.desktop().screenGeometry().height()
        self.width = QApplication.desktop().screenGeometry().width()
        self.path = 'TestImages/bus.jpg'
        self.savepath = 'runs'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.model = 'yolov5s.pt'
        self.setupUi(self)
        self.retranslateUi(self)
        self.terminal = terminalWindow.Ui_Dialog()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Object detection")
        MainWindow.resize(int(self.width * 3 / 4), int(self.height * 3 / 4))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 0, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 2, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 0, 3, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 706, 22))
        self.menubar.setObjectName("menubar")
        self.menuMain = QtWidgets.QMenu(self.menubar)
        self.menuMain.setObjectName("menuMain")
        self.menuSetting = QtWidgets.QMenu(self.menubar)
        self.menuSetting.setObjectName("menuSetting")
        self.menuDevice = QtWidgets.QMenu(self.menuSetting)
        self.menuDevice.setObjectName("menuDevice")
        self.menuModel = QtWidgets.QMenu(self.menuSetting)
        self.menuModel.setObjectName("menuModel")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_image = QtWidgets.QAction(MainWindow)
        self.actionOpen_image.setCheckable(False)
        self.actionOpen_image.setObjectName("actionOpen_image")
        self.actionOpen_webcam = QtWidgets.QAction(MainWindow)
        self.actionOpen_webcam.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionCPU = QtWidgets.QAction(MainWindow)
        self.actionCPU.setCheckable(True)
        self.actionCPU.setChecked(not torch.cuda.is_available())
        self.actionCPU.setObjectName("actionCPU")
        self.actionGPU = QtWidgets.QAction(MainWindow)
        self.actionGPU.setCheckable(True)
        self.actionGPU.setChecked(torch.cuda.is_available())
        self.actionGPU.setObjectName("actionGPU")
        self.actionOpen_folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_folder.setObjectName("actionOpen_folder")
        self.actionOpen_video = QtWidgets.QAction(MainWindow)
        self.actionOpen_video.setObjectName("actionOpen_video")
        self.actionTerminal = QtWidgets.QAction(MainWindow)
        self.actionTerminal.setObjectName("actionTerminal")
        self.actionyolov5s = QtWidgets.QAction(MainWindow)
        self.actionyolov5s.setCheckable(True)
        self.actionyolov5s.setChecked(True)
        self.actionyolov5s.setObjectName("actionyolov5s")
        self.actionyolov5m = QtWidgets.QAction(MainWindow)
        self.actionyolov5m.setCheckable(True)
        self.actionyolov5m.setObjectName("actionyolov5m")
        self.actionyolov5l = QtWidgets.QAction(MainWindow)
        self.actionyolov5l.setCheckable(True)
        self.actionyolov5l.setObjectName("actionyolov5l")
        self.actionyolov5x = QtWidgets.QAction(MainWindow)
        self.actionyolov5x.setCheckable(True)
        self.actionyolov5x.setObjectName("actionyolov5x")
        self.actionSaving_Path = QtWidgets.QAction(MainWindow)
        self.actionSaving_Path.setObjectName("actionSaving_Path")
        self.menuMain.addAction(self.actionOpen_image)
        self.menuMain.addAction(self.actionOpen_video)
        self.menuMain.addAction(self.actionOpen_folder)
        self.menuMain.addAction(self.actionOpen_webcam)
        self.menuMain.addSeparator()
        self.menuMain.addAction(self.actionExit)
        self.menuDevice.addAction(self.actionCPU)
        self.menuDevice.addAction(self.actionGPU)
        self.menuModel.addAction(self.actionyolov5s)
        self.menuModel.addAction(self.actionyolov5m)
        self.menuModel.addAction(self.actionyolov5l)
        self.menuModel.addAction(self.actionyolov5x)
        self.menuSetting.addAction(self.menuModel.menuAction())
        self.menuSetting.addAction(self.menuDevice.menuAction())
        self.menuSetting.addAction(self.actionSaving_Path)
        self.menuTools.addAction(self.actionTerminal)
        self.menubar.addAction(self.menuMain.menuAction())
        self.menubar.addAction(self.menuSetting.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.actionOpen_image.triggered.connect(self.open_img)
        self.actionOpen_video.triggered.connect(self.open_video)
        self.actionOpen_folder.triggered.connect(self.open_folder)
        self.pushButton.clicked.connect(self.run)
        self.pushButton_2.clicked.connect(self.change_save)
        self.actionCPU.triggered.connect(self.setCPU)
        self.actionGPU.triggered.connect(self.setGPU)
        self.actionyolov5s.triggered.connect(self.model_s)
        self.actionyolov5m.triggered.connect(self.model_m)
        self.actionyolov5l.triggered.connect(self.model_l)
        self.actionyolov5x.triggered.connect(self.model_x)
        self.terminal = terminalWindow.Ui_Dialog()
        self.actionTerminal.triggered.connect(self.show_terminal)
        self.pushButton_3.clicked.connect(self.open_result_folder)
        self.actionSaving_Path.triggered.connect(self.change_save)
        self.progress.connect(self.updateProgressBar)
        self.display_signal.connect(self.display)
        self.actionOpen_webcam.triggered.connect(self.open_webcam)
        self.actionExit.triggered.connect(self.stop_run)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Object detection", "Object detection"))
        self.pushButton_2.setText(_translate("Object detection", "Change saving path"))
        self.pushButton.setText(_translate("Object detection", "Run"))
        self.pushButton_3.setText(_translate("Object detection", "Result"))
        self.menuMain.setTitle(_translate("Object detection", "File"))
        self.menuSetting.setTitle(_translate("Object detection", "Setting"))
        self.menuDevice.setTitle(_translate("Object detection", "Device"))
        self.menuModel.setTitle(_translate("Object detection", "Model"))
        self.menuTools.setTitle(_translate("Object detection", "Tools"))
        self.actionOpen_image.setText(_translate("Object detection", "Open image"))
        self.actionOpen_webcam.setText(_translate("Object detection", "Open webcam"))
        self.actionExit.setText(_translate("Object detection", "Exit"))
        self.actionCPU.setText(_translate("Object detection", "CPU"))
        self.actionGPU.setText(_translate("Object detection", "GPU"))
        self.actionOpen_folder.setText(_translate("Object detection", "Open folder"))
        self.actionOpen_video.setText(_translate("Object detection", "Open video"))
        self.actionTerminal.setText(_translate("Object detection", "Terminal"))
        self.actionyolov5s.setText(_translate("Object detection", "yolov5s"))
        self.actionyolov5m.setText(_translate("Object detection", "yolov5m"))
        self.actionyolov5l.setText(_translate("Object detection", "yolov5l"))
        self.actionyolov5x.setText(_translate("Object detection", "yolov5x"))
        self.actionSaving_Path.setText(_translate("Object detection", "Saving Path"))

    def updateProgressBar(self, val):
        self.progressBar.setValue(val)

    def open_img(self):
        file_name = QFileDialog.getOpenFileName(self, 'Choose file', './', 'image(*.jpg , *.png)')
        image_path = file_name[0]
        self.path = image_path
        self.display_signal.emit(self.path)

    def open_video(self):
        file_name = QFileDialog.getOpenFileName(self, 'Choose file', './', 'video(*.mp4)')
        self.path = file_name[0]
        print(self.path)
        self.display_signal.emit(self.path)

    def open_folder(self):
        p = QFileDialog.getExistingDirectory(self, 'Choose folder')
        # Check file type inside folder
        for files in os.listdir(p):
            if not files.endswith(".mp4") and not files.endswith(".jpg") and not files.endswith(".png"):
                # show error message and return
                s = "errorMessage/ERROR_1.jpg"
                self.display_signal.emit(s)
                return
        # show the first img or first frame of the first video inside the directory
        self.path = p
        first_file = os.listdir(self.path)[0]
        s = self.path + '/' + first_file
        self.display_signal.emit(s)

    def open_webcam(self):
        self.path = '0'

    def display(self, path):
        if path.endswith(".jpg") or path.endswith(".png") or path.endswith(".jpeg"):
            img = cv2.imread(path)
            show_img(img, self)
        # show video
        elif path.endswith(".mp4"):
            # self.video_play(path)
            capture = cv2.VideoCapture(path)
            if capture.isOpened():
                # while True:
                ret, img = capture.read()
                # if not ret:
                # break
                show_img(img, self)
                # time.sleep(0.2)
            else:
                print('video open fail')
            # TODO: show each frame while running (optional)

    def change_save(self):
        self.savepath = QFileDialog.getExistingDirectory(self, 'Choose save path')

    def run(self):
        self.run_thread = Run(self)
        self.run_thread.start()

    def stop_run(self):
        self.run_thread.terminate()
        # self.run_thread.stop()

    def setCPU(self):
        self.device = 'cpu'
        self.actionGPU.setChecked(False)

    def setGPU(self):
        self.device = '0'
        self.actionCPU.setChecked(False)

    def model_s(self):
        self.model = 'yolov5s.pt'
        self.actionyolov5m.setChecked(False)
        self.actionyolov5l.setChecked(False)
        self.actionyolov5x.setChecked(False)

    def model_m(self):
        self.model = 'yolov5m.pt'
        self.actionyolov5s.setChecked(False)
        self.actionyolov5l.setChecked(False)
        self.actionyolov5x.setChecked(False)

    def model_l(self):
        self.model = 'yolov5l.pt'
        self.actionyolov5m.setChecked(False)
        self.actionyolov5s.setChecked(False)
        self.actionyolov5x.setChecked(False)

    def model_x(self):
        self.model = 'yolov5x.pt'
        self.actionyolov5m.setChecked(False)
        self.actionyolov5l.setChecked(False)
        self.actionyolov5s.setChecked(False)

    def show_terminal(self):
        self.terminal.show()

    def open_result_folder(self):
        os.startfile(self.savepath)

    def video_play(self, path):
        self.video_play_thread = Video_play(self, path)
        self.video_play_thread.start()


class Run(QThread):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        # self._stop_event = threading.Event()

    # def stop(self):
    #     self._stop_event.set()

    def run(self):
        # get save path
        p = Path(self.main_window.path)
        # run detect
        if self.main_window == '0':
            detect.run(window=self.main_window, source=self.main_window.path, weights=self.main_window.model,
                       device=self.main_window.device, project=self.main_window.savepath, nosave=True)
        else:
            detect.run(window=self.main_window, source=self.main_window.path, weights=self.main_window.model,
                       device=self.main_window.device, project=self.main_window.savepath)
        if os.path.isdir(p):
            # show the first img or first frame of the first video in a directory
            first_file = os.listdir(self.main_window.path)[0]
            s = self.main_window.savepath + '/' + first_file
        else:
            s = self.main_window.savepath + '/' + p.name
        self.main_window.display_signal.emit(s)



class Video_play(QThread):
    def __init__(self, main_window, path):
        super().__init__()
        self.main_window = main_window
        self.path = path

    def run(self):
        capture = cv2.VideoCapture(self.path)
        if capture.isOpened():
            while True:
                ret, img = capture.read()
                if not ret:
                    break
                self.main_window.next_frame.emit(img)
                time.sleep(0.1)


def show_img(img, window):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
    x = img.shape[1]  # 获取图像大小
    y = img.shape[0]
    ratio = x / y
    window_x = window.graphicsView.width() - 10
    window_y = window.graphicsView.height() - 10
    if ratio * window_y <= window_x:
        x = int(ratio * window_y)
        y = window_y
    else:
        x = window_x
        y = int(window_x / ratio)
    img = cv2.resize(img, (x, y))
    frame = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
    pix = QPixmap.fromImage(frame)
    window.item = QGraphicsPixmapItem(pix)
    window.scene = QGraphicsScene()
    window.scene.addItem(window.item)
    window.graphicsView.setScene(window.scene)


def play_video(path):
    pass
