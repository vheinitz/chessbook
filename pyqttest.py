import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow

Ui_MainWindow, BaseClass = uic.loadUiType("test.ui")


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        # Connect the button click signal to the slot
        self.b.clicked.connect(self.toggle_checkbox)

    def toggle_checkbox(self):
        # Toggle the checkbox state
        self.cb.setChecked(not self.cb.isChecked())


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
