"""
Contains colorful buttons, color schemes of which are derived from Bootstrap.
"""

from PySide6.QtWidgets import QPushButton, QWidget


class InfoButton(QPushButton):
    """
    Button widget that uses Bootstrap's info color.
    """
    def __init__(self, text: str, parent: QWidget = None):
        """
        Initializes the widget.
        :param text: Text to be displayed.
        :param parent: Parent widget.
        """
        super().__init__(text=text, parent=parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
            }
            QPushButton:pressed {
                background-color: #138496;
            }
            QPushButton:disabled {
                background-color: #555555;
                border: 2px inset #17a2b8;
                color: white;
            }
        """)


class SuccessButton(QPushButton):
    """
    Button widget that uses Bootstrap's success color
    """
    def __init__(self, text: str,  parent: QWidget = None):
        """
        Initializes the widget.
        :param text: Text to be displayed.
        :param parent: Parent widget.
        """
        super().__init__(text=text, parent=parent)

        self.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
            }
            QPushButton:pressed {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #555555;
                border: 2px inset #28a745;
                color: white;
            }
        """)


class DangerButton(QPushButton):
    """
    Button widget that uses Bootstrap's danger color.
    """
    def __init__(self, text: str,  parent: QWidget = None):
        """
        Initializes the widget.
        :param text: Text to be displayed.
        :param parent: Parent widget.
        """
        super().__init__(text=text, parent=parent)

        self.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
            }
            QPushButton:pressed {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #555555;
                border: 2px inset #dc3545;
                color: white;
            }
        """)


class WarningButton(QPushButton):
    """
    Button widget that uses Bootstrap's warning color.
    """
    def __init__(self, text: str,  parent: QWidget = None):
        """
        Initializes the widget.
        :param text: Text to be displayed.
        :param parent: Parent widget.
        """
        super().__init__(text=text, parent=parent)

        self.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: black;
                border: none;
            }
            QPushButton:pressed {
                background-color: #e0a800;
            }
            QPushButton:disabled {
                background-color: #555555;
                border: 2px inset #ffc107;
                color: white;
            }
        """)


class DarkWarningButton(QPushButton):
    """
    Button widget that uses Bootstrap's warning color, except slightly darker.
    """
    def __init__(self, text: str,  parent: QWidget = None):
        """
        Initializes the widget.
        :param text: Text to be displayed.
        :param parent: Parent widget.
        """
        super().__init__(text=text, parent=parent)

        self.setStyleSheet("""
            QPushButton {
                background-color: #d39e00;
                color: white;
                border: none;
            }
            QPushButton:pressed {
                background-color: #b38600;
            }
            QPushButton:disabled {
                background-color: #555555;
                border: 2px inset #d39e00;
                color: white;
            }
        """)
