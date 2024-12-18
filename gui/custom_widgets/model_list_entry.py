import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QFont
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout
from sympy.strategies import typed

from shared import constants


def _create_gradient_pixmap() -> QPixmap:
    """
    Creates a 16x16 RGB gradient icon as a QPixmap.
    :return: A gradient QPixmap.
    """
    pixmap = QPixmap(16, 16)
    painter = QPainter(pixmap)

    # Define a linear gradient
    gradient = QLinearGradient(0, 0, 16, 16)
    gradient.setColorAt(0, QColor(255, 0, 0))  # Red
    gradient.setColorAt(0.5, QColor(0, 255, 0))  # Green
    gradient.setColorAt(1, QColor(0, 0, 255))  # Blue

    # Fill the pixmap with the gradient
    painter.fillRect(0, 0, 16, 16, gradient)
    painter.end()
    return pixmap

def _create_rgb_squares_pixmap() -> QPixmap:
    """
    Creates a 16x16 RGB strips icon as a QPixmap.
    :return: RGB-Stripes as a QPixmap.
    """

    pixmap = QPixmap(16, 16)
    painter = QPainter(pixmap)

    # Draw red square
    painter.fillRect(0, 0, 5, 16, QColor(255, 0, 0))

    # Draw green square
    painter.fillRect(5, 0, 5, 16, QColor(0, 255, 0))

    # Draw blue square
    painter.fillRect(10, 0, 6, 16, QColor(0, 0, 255))

    painter.end()
    return pixmap

class ModelListEntry(QWidget):
    """
    A QListWidget entry widget that is supposed to be a user-friendly representation of a model on a list.
    """
    def __init__(self, model_path: str, parent=None):
        """
        Initializes the widget with a path to a model, from which the relevant information will be extracted.
        :param model_path: String path to a model.
        :param parent: Parent widget.
        """
        super(ModelListEntry, self).__init__(parent)

        # Extract information from the path to a model.
        self._path_to_model = model_path
        mode, name, epochs, is_checkpoint = ModelListEntry._parse_filename(model_path)

        # Build UI
        layout = QHBoxLayout(self)

        icon_label = QLabel()
        if mode == constants.ModelCategory.GSC:
            icon_label.setPixmap(_create_rgb_squares_pixmap().scaledToHeight(32))
        else:
            icon_label.setPixmap(_create_gradient_pixmap().scaledToHeight(32))
        layout.addWidget(icon_label)

        text_layout = QVBoxLayout()
        layout.addLayout(text_layout)
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Upper text (bold)
        upper_label = QLabel(name)
        font = QFont()
        font.setBold(True)
        upper_label.setFont(font)
        text_layout.addWidget(upper_label)

        # Lower text (regular)
        lower_label = QLabel(f"Epochs: {epochs}, Mode: {mode.name}" + (" | (checkpoint)" if is_checkpoint else ""))
        text_layout.addWidget(lower_label)

        # Align the layout properly
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)


    @property
    def model_path(self):
        return self._path_to_model

    @staticmethod
    def _parse_filename(filename: str) -> tuple[constants.ModelCategory, str, int, bool] | None:
        filename = Path(filename).name
        pattern = r"(gsc|rgb)_(.*)_(\d*)e\.pth"
        pattern_match = re.match(pattern, filename)
        if pattern_match:
            mode_str = pattern_match.group(1)
            mode = constants.ModelCategory.GSC if mode_str == "gsc" else constants.ModelCategory.RGB
            name = pattern_match.group(2)
            epochs = int(pattern_match.group(3))
            is_checkpoint = "-cp" in name
            return mode, name, epochs, is_checkpoint
        return None
