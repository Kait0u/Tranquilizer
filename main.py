import qdarktheme
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication
from gui.app import MainWindow
from shared.init_activities import initialization_activities


def main():
    initialization_activities()

    app = QApplication([])

    qdarktheme.setup_theme("dark")
    qdarktheme.enable_hi_dpi()

    window = MainWindow()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
