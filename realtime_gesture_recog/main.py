import sys
import threading

import cv2
import torch
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
from qtpy import QtCore, QtWidgets, QtGui

import global_vars
import models
from filters import skinMask
from main_ui import Ui_MainWindow
from models import load_model
from utils import *


class MainWindow(QMainWindow, Ui_MainWindow):
    # define signals
    signal_gesture_read = pyqtSignal(name="signal_gesture_read")
    signal_prediction_show = pyqtSignal(name="signal_prediction_show")

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # CNN
        self.model = load_model()
        if torch.cuda.is_available():
            self.gpu = True
            self.model.cuda()
        self.prediction_frequency = 10  # each 10 images arise a prediction
        self.prediction_count = 0

        # camera
        self.camera = cv2.VideoCapture()
        self.timer_camera = QtCore.QTimer()
        self.CAM_NUM = 0
        self.camera_x0 = 400
        self.camera_y0 = 200
        self.camera_height = 200
        self.camera_width = 200

        # control
        self.horizontalSlider.setValue(70)
        self.threshold = self.horizontalSlider.value()

        # SLOT
        self.slot_init()

    @log_decorator
    def slot_init(self):
        self.pushButton_open_cam.clicked.connect(self.camera_open)
        self.pushButton_close_cam.clicked.connect(self.camera_close)
        self.timer_camera.timeout.connect(self.camera_show)
        self.pushButton_left.clicked.connect(self.box_left)
        self.pushButton_right.clicked.connect(self.box_right)
        self.pushButton_down.clicked.connect(self.box_down)
        self.pushButton_up.clicked.connect(self.box_up)
        self.horizontalSlider.valueChanged.connect(self.threshold_update)

        self.signal_prediction_show.connect(self.prediction_show)
        self.signal_gesture_read.connect(self.gesture_show)

    @log_decorator
    def camera_open(self, *args):
        '''
        open laptop camera
        :param args:
        :return:
        '''
        if self.timer_camera.isActive() == False:
            flag = self.camera.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)

    @log_decorator
    def camera_close(self, *args):
        '''
        close laptop camera
        :param args:
        :return:
        '''
        self.timer_camera.stop()
        self.camera.release()
        self.label_camera.clear()
        self.label_camera.setText("Not recieve car video stream yet.")

    def camera_show(self):
        '''
        show camera image, emit signal_gesture_read signal
        :return:
        '''
        flag, frame = self.camera.read()
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (640, 480))
        self.image = frame

        threshold_str = "Current threshlold: %d %%" % self.threshold
        cv2.putText(frame, 'Driver Romm <( ^-^ )>', (FX, FY), FONT, SIZE, (0, 255, 0), 1, 1)
        cv2.putText(frame, threshold_str, (FX, FY + 2 * FH), FONT, SIZE, (0, 255, 0), 1, 1)
        cv2.rectangle(frame, (self.camera_x0, self.camera_y0),
                      (self.camera_x0 + self.camera_width, self.camera_y0 + self.camera_height), (0, 255, 0), 1)

        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # call gesture filter
        self.signal_gesture_read.emit()

    def gesture_show(self):
        '''
        show gesture image after binary mask, emit signal_prediction_show signal
        :return:
        '''
        roi = skinMask(self.image, self.camera_x0, self.camera_y0, self.camera_width, self.camera_height)
        showImage = QtGui.QImage(roi.data, roi.shape[1], roi.shape[0], QtGui.QImage.Format_Grayscale8)
        self.label_gesture.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # predict image
        if self.prediction_count == self.prediction_frequency:
            t = threading.Thread(target=models.predict_gesture, args=[self.model, roi],kwargs={"verbose":False})
            t.start()
            self.prediction_count = 0
        else:
            self.prediction_count += 1
        self.signal_prediction_show.emit()

    def prediction_show(self):
        '''
        show prediction.
        :return:
        '''
        # draw probability plot
        plot = models.update()
        # write result
        if len(global_vars.jsonarray) > 0:
            prediction = max(global_vars.jsonarray, key=lambda x: global_vars.jsonarray[x])
            if global_vars.jsonarray[prediction] * 100 > self.threshold:
                self.label_predict.setText(GESTURE_CLASSES[int(prediction)])
        else:
            self.label_predict.setText('nothing')
        showImage = QtGui.QImage(plot.data, plot.shape[1], plot.shape[0], QtGui.QImage.Format_RGB888)
        self.label_probability.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # SLOT: update value
    def box_left(self):
        self.camera_x0 -= 5

    def box_right(self):
        self.camera_x0 += 5

    def box_up(self):
        self.camera_y0 -= 5

    def box_down(self):
        self.camera_y0 += 5

    def threshold_update(self):
        self.threshold = self.horizontalSlider.value()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    # ui.showFullScreen()
    sys.exit(app.exec_())
