# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 836)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_close_cam = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close_cam.setObjectName("pushButton_close_cam")
        self.gridLayout.addWidget(self.pushButton_close_cam, 2, 1, 1, 1)
        self.pushButton_connect_car = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_connect_car.setObjectName("pushButton_connect_car")
        self.gridLayout.addWidget(self.pushButton_connect_car, 7, 0, 1, 1)
        self.label_threshold = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold.setObjectName("label_threshold")
        self.gridLayout.addWidget(self.label_threshold, 5, 0, 1, 1)
        self.frame_result = QtWidgets.QFrame(self.centralwidget)
        self.frame_result.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_result.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_result.setObjectName("frame_result")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_result)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_result = QtWidgets.QLabel(self.frame_result)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_result.sizePolicy().hasHeightForWidth())
        self.label_result.setSizePolicy(sizePolicy)
        self.label_result.setObjectName("label_result")
        self.gridLayout_7.addWidget(self.label_result, 0, 0, 1, 1)
        self.label_predict = QtWidgets.QLabel(self.frame_result)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_predict.sizePolicy().hasHeightForWidth())
        self.label_predict.setSizePolicy(sizePolicy)
        self.label_predict.setObjectName("label_predict")
        self.gridLayout_7.addWidget(self.label_predict, 0, 1, 1, 1)
        self.label_probability = QtWidgets.QLabel(self.frame_result)
        self.label_probability.setObjectName("label_probability")
        self.gridLayout_7.addWidget(self.label_probability, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.frame_result, 8, 0, 1, 2)
        self.pushButton_down = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_down.setObjectName("pushButton_down")
        self.gridLayout.addWidget(self.pushButton_down, 4, 1, 1, 1)
        self.frame_gesture = QtWidgets.QFrame(self.centralwidget)
        self.frame_gesture.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_gesture.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_gesture.setObjectName("frame_gesture")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_gesture)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_gesture = QtWidgets.QLabel(self.frame_gesture)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_gesture.sizePolicy().hasHeightForWidth())
        self.label_gesture.setSizePolicy(sizePolicy)
        self.label_gesture.setAlignment(QtCore.Qt.AlignCenter)
        self.label_gesture.setObjectName("label_gesture")
        self.gridLayout_6.addWidget(self.label_gesture, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame_gesture, 0, 0, 1, 2)
        self.pushButton_up = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_up.setObjectName("pushButton_up")
        self.gridLayout.addWidget(self.pushButton_up, 4, 0, 1, 1)
        self.pushButton_open_cam = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open_cam.setObjectName("pushButton_open_cam")
        self.gridLayout.addWidget(self.pushButton_open_cam, 2, 0, 1, 1)
        self.pushButton_right = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_right.setObjectName("pushButton_right")
        self.gridLayout.addWidget(self.pushButton_right, 3, 1, 1, 1)
        self.pushButton_disconnect = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_disconnect.setObjectName("pushButton_disconnect")
        self.gridLayout.addWidget(self.pushButton_disconnect, 7, 1, 1, 1)
        self.pushButton_left = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_left.setObjectName("pushButton_left")
        self.gridLayout.addWidget(self.pushButton_left, 3, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout.addItem(spacerItem, 6, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout.addItem(spacerItem1, 6, 0, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider.sizePolicy().hasHeightForWidth())
        self.horizontalSlider.setSizePolicy(sizePolicy)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 5, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 1, 1, 1)
        self.frame_camera = QtWidgets.QFrame(self.centralwidget)
        self.frame_camera.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_camera.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_camera.setObjectName("frame_camera")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_camera)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_human = QtWidgets.QFrame(self.frame_camera)
        self.frame_human.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_human.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_human.setObjectName("frame_human")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_human)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_camera = QtWidgets.QLabel(self.frame_human)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_camera.sizePolicy().hasHeightForWidth())
        self.label_camera.setSizePolicy(sizePolicy)
        self.label_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.label_camera.setObjectName("label_camera")
        self.gridLayout_5.addWidget(self.label_camera, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_human, 0, 0, 1, 1)
        self.frame_car = QtWidgets.QFrame(self.frame_camera)
        self.frame_car.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_car.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_car.setObjectName("frame_car")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_car)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_car = QtWidgets.QLabel(self.frame_car)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_car.sizePolicy().hasHeightForWidth())
        self.label_car.setSizePolicy(sizePolicy)
        self.label_car.setAlignment(QtCore.Qt.AlignCenter)
        self.label_car.setObjectName("label_car")
        self.gridLayout_4.addWidget(self.label_car, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_car, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame_camera, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RemoteCarContorl"))
        self.pushButton_close_cam.setText(_translate("MainWindow", "close camera"))
        self.pushButton_connect_car.setText(_translate("MainWindow", "connect car"))
        self.label_threshold.setText(_translate("MainWindow", "detection threshold"))
        self.label_result.setText(_translate("MainWindow", "Prediction:"))
        self.label_predict.setText(_translate("MainWindow", "Nothing"))
        self.label_probability.setText(_translate("MainWindow", "Probability"))
        self.pushButton_down.setText(_translate("MainWindow", "down"))
        self.label_gesture.setText(_translate("MainWindow", "gesture capture."))
        self.pushButton_up.setText(_translate("MainWindow", "up"))
        self.pushButton_open_cam.setText(_translate("MainWindow", "open camera"))
        self.pushButton_right.setText(_translate("MainWindow", "right"))
        self.pushButton_disconnect.setText(_translate("MainWindow", "disconnect"))
        self.pushButton_left.setText(_translate("MainWindow", "left"))
        self.label_camera.setText(_translate("MainWindow", "Not open camera yet."))
        self.label_car.setText(_translate("MainWindow", "Not recieve car video stream yet."))
