from PIL import Image, ImageQt
from PySide6.QtGui import QPixmap

def pil_to_qt(pil_image: Image.Image) -> QPixmap:
    """
    Converts a PIL image to a QPixmap.
    :param pil_image: A PIL Image.
    :return: A QPixmap made out of the provided PIL Image.
    """

    qim = ImageQt.ImageQt(pil_image)
    pix = QPixmap.fromImage(qim)
    return pix

