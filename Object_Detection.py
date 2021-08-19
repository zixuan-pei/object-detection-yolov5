import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from GUI.mainWindow import Ui_MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ObjectDetection = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(ObjectDetection)
    ObjectDetection.show()
    sys.exit(app.exec_())
