from enum import Enum
from pathlib import Path


class ModelCategory(Enum):
    """
    Contains the model category (mode) names.
    """
    RGB = "RGB"
    GSC = "GSC (CHANNEL SPLIT)"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        return list(map(lambda c: c, cls))

APP_NAME = "Tranquilizer"
COPYRIGHT_MESSAGE = "By Jakub Jaworski (Kait0u), October 2024"

MODELS_DIRECTORY_PATH: Path = Path("./models")
ICON_PATH: Path = Path("./assets/icon.png")

# Initialization constants
DIRS_TO_ENSURE = [
    "./models/rgb",
    "./models/gsc",
]

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600


