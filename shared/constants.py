from enum import Enum
from pathlib import Path

from PIL.TiffImagePlugin import COPYRIGHT


class ModelCategory(Enum):
    RGB = "RGB"
    GSC = "GSC (CHANNEL SPLIT)"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        return list(map(lambda c: c, cls))

COPYRIGHT_MESSAGE = "By Jakub Jaworski (Kait0u), October 2024"

MODELS_DIRECTORY_PATH: Path = Path("./models")
ICON_PATH: Path = Path("./assets/icon.png")

# Initialization constants
DIRS_TO_ENSURE = [
    "./models/rgb",
    "./models/gsc",
]


