import numpy as np
from PIL import Image, ImageQt
from PySide6.QtGui import QPixmap


def maxmin_scale(arr: np.ndarray, new_min = 0, new_max = 1) -> np.ndarray:
    old_min = np.min(arr)
    old_max = np.max(arr)

    # Rescaling formula applied
    scaled = (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return scaled

def pil_to_qt(pil_image: Image.Image) -> QPixmap:
    qim = ImageQt.ImageQt(pil_image)
    pix = QPixmap.fromImage(qim)
    return pix

