from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class AspectRatioLabel(QLabel):
    """
    An image label which will attempt to maintain an aspect ratio.
    """
    def __init__(self, pixmap = None):
        super().__init__()
        self.pixmap = None

        if pixmap is not None:
            self.setPixmap(pixmap)

    def setPixmap(self, pixmap, internal=False):
        super().setPixmap(pixmap)
        if not internal:
            self.pixmap = pixmap

    def resizeEvent(self, event):
        if self.pixmap is None:
            super().resizeEvent(event)
            return

        # Get the current size of the label
        label_size = event.size()

        # Get the original size of the pixmap
        original_pixmap_size = self.pixmap.size()

        # Calculate the aspect ratio
        aspect_ratio = original_pixmap_size.width() / original_pixmap_size.height()

        # Calculate new size while maintaining aspect ratio
        if label_size.width() / label_size.height() > aspect_ratio:
            new_width = label_size.height() * aspect_ratio
            new_height = label_size.height()
        else:
            new_width = label_size.width()
            new_height = label_size.width() / aspect_ratio

        # Resize the pixmap and set it
        self.setPixmap(self.pixmap.scaled(new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation), internal=True)
        super().resizeEvent(event)