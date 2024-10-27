import os
from pathlib import Path

from shared import constants


def initialization_activities() -> None:
    """
    Performs the necessary activities that prepare the program's environment.
    """

    # Ensure necessary directories exist
    dirs_to_ensure = constants.DIRS_TO_ENSURE
    for dirpath in dirs_to_ensure:
        __ensure_directory(dirpath)


def __ensure_directory(path: str) -> None:
    """
    Ensures that a directory exists. If it does not - it will be created.
    :param path: A string with a path to a directory.
    """

    # Convert the path to a Path object for easier manipulation
    directory = Path(path)

    # Check if it's a directory
    if not directory.is_dir():
        # If it's not a directory, create it (including any parent directories)
        directory.mkdir(parents=True, exist_ok=True)