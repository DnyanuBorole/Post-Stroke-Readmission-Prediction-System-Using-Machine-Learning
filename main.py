# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\QT\strokes predictor\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import background
from prediction import prdct
from classifcation import classfy
from performance import view_data

class Ui_Dialog(object):
    def prd(self):
        self.Dialog = QtWidgets.QDialog()
        self.ui = prdct()
        self.ui.setupUi(self.Dialog)
        self.Dialog.show()
    def classify_(self):
        classfy()
        self.Dialog = QtWidgets.QDialog()
        self.ui = view_data()
        self.ui.setupUi(self.Dialog)
        self.ui.performance()
        self.Dialog.show()
        # self.Dialog.show()


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1024, 670)
        Dialog.setStyleSheet("QDialog{background-image: url(:/dialog/dialog.jpg);}")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(440, 132, 351, 391))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label.setLineWidth(4)
        self.label.setMidLineWidth(0)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(490, 320, 271, 51))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 105, 29, 255), stop:1 rgba(255, 255, 255, 255));\n"
"font: 87 15pt \"Arial Black\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(490, 420, 271, 51))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 105, 29, 255), stop:1 rgba(255, 255, 255, 255));\n"
"font: 87 12pt \"Arial Black\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(450, 140, 131, 121))
        self.label_2.setStyleSheet("background-image: url(:/dialog/dr.png);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(80, 20, 820, 60))
        self.label_3.setStyleSheet("font: 22pt \"Baskerville Old Face\";\n"
                                   "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 255, 255, 160), stop:1 rgba(255, 255, 255, 255));\n"
                                   "\n"
                                   "")
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_3.setLineWidth(2)
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.pushButton.clicked.connect(self.prd)
        self.pushButton_2.clicked.connect(self.classify_)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Prediction"))
        self.pushButton_2.setText(_translate("Dialog", "Algorithms Performance"))
        self.label_3.setText(_translate("Dialog", "Post Stroke Readmission Prediction System Using ML"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
