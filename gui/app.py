from pathlib import Path

import torch
from PIL import Image
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QColorConstants, QScreen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QLabel, QFileDialog, QFrame, QMainWindow,
    QComboBox, QLineEdit,
    QToolButton, QTabWidget, QScrollArea, QGroupBox, QStatusBar, QMessageBox, QApplication
)

from backend.denoise import denoise
from backend.net import DnCNN
from backend.utils import pil_to_qt
from gui.custom_widgets.buttons import SuccessButton, InfoButton
from gui.custom_widgets.image_labels import AspectRatioLabel
from gui.custom_widgets.model_list_entry import ModelListEntry
from shared import constants


class MainWindow(QMainWindow):
    """
    The main window of the GUI application.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle(constants.APP_NAME)
        self.setGeometry(0, 0, constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT)
        self.setMinimumSize(QSize(constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT))

        # Center the window
        center = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

        # Set the icon
        self.setWindowIcon(QIcon(str(constants.ICON_PATH)))

        # Build the UI and setup its hooks
        self.build_ui()
        self._fill_data()
        self.setup_callbacks()

        # Data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = constants.ModelCategory.RGB
        self.selected_model_path = ""
        self.selected_image_path = ""

        # Initialize other variables (mostly state variables)
        self.model = None
        self.model_name = None
        self.model_mode = None
        self.noisy_image = None
        self.denoised_image = None
        self.details_image = None

        # Final setups
        self._update_form()

    def build_ui(self) -> None:
        """
        Builds the UI of the app.
        """

        # Status bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        right_label = QLabel(constants.COPYRIGHT_MESSAGE)
        self.status_bar.addPermanentWidget(right_label)

        # Body
        body = QWidget()
        self.setCentralWidget(body)
        layout_body = QHBoxLayout(body)

        # Left side bar
        sidebar_left = QWidget()
        sidebar_left.setMinimumWidth(250)
        sidebar_left.setMaximumWidth(300)
        layout_sidebar_left = QVBoxLayout(sidebar_left)

        self.combobox_model_group = QComboBox()
        layout_sidebar_left.addWidget(self.combobox_model_group)

        self.list_models = QListWidget()
        layout_sidebar_left.addWidget(self.list_models)

        self.btn_load = InfoButton("Load")
        layout_sidebar_left.addWidget(self.btn_load)

        vsep = QFrame()
        vsep.setFrameStyle(QFrame.Shape.VLine)

        # Main widget (right)
        widget_right = QWidget()
        layout_widget_right = QVBoxLayout(widget_right)

        # Input file bar
        widget_file_bar = QWidget()
        layout_widget_file_bar = QHBoxLayout(widget_file_bar)

        layout_widget_file_bar.addWidget(QLabel("Noisy Image: "))

        self.tf_file_path = QLineEdit()
        self.tf_file_path.setReadOnly(True)
        layout_widget_file_bar.addWidget(self.tf_file_path)

        self.btn_browse = QToolButton()
        self.btn_browse.setIcon(QIcon.fromTheme("document-open"))
        layout_widget_file_bar.addWidget(self.btn_browse)

        self.btn_start = SuccessButton("Denoise")
        self.btn_start.setMinimumWidth(75)
        layout_widget_file_bar.addWidget(self.btn_start)

        layout_widget_right.addWidget(widget_file_bar)

        # The result tabs
        self.tabwidget = QTabWidget()
        self._init_tabs()
        layout_widget_right.addWidget(self.tabwidget)

        layout_body.addWidget(sidebar_left)
        layout_body.addWidget(vsep)
        layout_body.addWidget(widget_right)

    def _init_tabs(self) -> None:
        """
        Initializes the output tabs.
        """

        # View 1 - Direct
        view1 = QWidget()
        layout_view1 = QVBoxLayout(view1)

        # Input image
        groubox_loaded = QGroupBox("Loaded Image")
        layout_groupbox_loaded = QHBoxLayout(groubox_loaded)
        layout_groupbox_loaded.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.label_img_loaded = AspectRatioLabel()
        layout_groupbox_loaded.addWidget(self.label_img_loaded)
        layout_view1.addWidget(groubox_loaded)

        # Denoised image
        groupbox_denoised = QGroupBox("Denoised Image")
        layout_groupbox_denoised = QVBoxLayout(groupbox_denoised)
        layout_groupbox_denoised.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.label_img_denoised = AspectRatioLabel()
        layout_groupbox_denoised.addWidget(self.label_img_denoised)
        layout_view1.addWidget(groupbox_denoised)

        self.btn_export_denoised = InfoButton(" Export Denoised Image")
        self.btn_export_denoised.setIcon(QIcon.fromTheme("document-save"))
        layout_view1.addWidget(self.btn_export_denoised)

        self.tabwidget.addTab(view1, "View 1 (Direct)")

        # View 2 - Matplotlib output
        view2 = QWidget()
        layout_view2 = QVBoxLayout(view2)

        # Matplotlib image
        self.label_mpl_summary = QLabel()
        self.label_mpl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.label_mpl_summary)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_area.setWidgetResizable(True)
        layout_view2.addWidget(scroll_area)

        self.btn_export_details = InfoButton("Export Details")
        self.btn_export_details.setIcon(QIcon.fromTheme("document-save"))
        layout_view2.addWidget(self.btn_export_details)

        self.tabwidget.addTab(view2, "View 2 (Process Summary)")

    def _fill_data(self) -> None:
        """
        Fills the widgets with necessary data.
        """
        self.combobox_model_group.addItems([cat.value for cat in constants.ModelCategory])
        self.mode = constants.ModelCategory.list()[self.combobox_model_group.currentIndex()]

        self._set_fallback_images()
        self._update_models()

    def _set_fallback_images(self) -> None:
        """
        Sets up some empty placeholder images and sets them where necessary.
        """
        pixmap_horizontal = QPixmap(160, 90)
        pixmap_horizontal.fill(QColorConstants.Transparent)
        pixmap_vertical = QPixmap(90, 160)
        pixmap_vertical.fill(QColorConstants.Transparent)

        self.label_img_loaded.setPixmap(pixmap_horizontal)
        self.label_img_denoised.setPixmap(pixmap_horizontal)
        self.label_mpl_summary.setPixmap(pixmap_vertical)

    def setup_callbacks(self) -> None:
        """
        Attaches callbacks to relevant widgets.
        """

        self.combobox_model_group.currentIndexChanged.connect(self._on_combobox_model_group_current_index_changed)
        self.list_models.currentItemChanged.connect(self._on_list_models_current_item_changed)
        self.btn_load.clicked.connect(self._on_btn_load_clicked)
        self.btn_browse.clicked.connect(self._on_btn_browse_clicked)
        self.btn_start.clicked.connect(self._on_btn_start_clicked)
        self.btn_export_denoised.clicked.connect(self._on_btn_export_denoised_clicked)
        self.btn_export_details.clicked.connect(self._on_btn_export_details_clicked)

    def _scan_for_models(self) -> list[Path]:
        """
        Scans the app's subdirectory (according to the selected mode) for models and returns a list of paths to them.
        :return: A list of Path objects that point to model files.
        """
        subdir = "gsc" if self.mode is constants.ModelCategory.GSC else "rgb"
        path_to_dir = constants.MODELS_DIRECTORY_PATH / subdir
        files = []
        for f in path_to_dir.iterdir():
            if f.is_file() and f.name.endswith(".pth"):
                files.append(f)
        files.sort(key=lambda model_file: model_file.stat().st_mtime, reverse=True)
        return files

    def _load_model(self) -> None:
        """
        Attempts to load a currently selected model. If something goes wrong, clears the session model.
        """
        try:
            model_mode = self.mode
            ch = 1 if model_mode == constants.ModelCategory.GSC else 3
            model = DnCNN(in_channels=ch, out_channels=ch).to(self.device)
            model.load_state_dict(torch.load(self.selected_model_path))
            model_name = Path(self.selected_model_path).name or None
        except Exception as e:
            model_mode = None
            model = None
            model_name = None
        self.model_mode = model_mode
        self.model = model
        self.model_name = model_name
        self._update_status()

    def _update_status(self) -> None:
        """
        Updates the status message.
        """
        self.status_bar.showMessage(self._generate_status_message())

    def _generate_status_message(self) -> str:
        """
        Generates a new status message that includes the currently loaded model.
        :return: A new status message.
        """
        return f"Current model: {self.model_name} ({self.device})"

    def _update_models(self) -> None:
        """
        Updates the model list with currently selected model types that are available at the time of calling.
        """

        files = self._scan_for_models()
        self.list_models.clear()
        self.selected_model_path = ""
        for file in files:
            path = str(file)
            item = QListWidgetItem(self.list_models)
            widget = ModelListEntry(path)
            w_size = widget.sizeHint()
            w_size.setHeight(56)
            item.setSizeHint(w_size)
            self.list_models.addItem(item)
            self.list_models.setItemWidget(item, widget)

    def _open_file(self) -> None:
        """
        Opens an image file using a file browser.
        Loads it into a relevant variable.
        """

        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpeg *.jpg *.bmp)")
        if file_path:
            self.noisy_image = Image.open(file_path)
            pixmap = QPixmap(file_path)
            self.label_img_loaded.setPixmap(pixmap)
            self.tf_file_path.setText(file_path)

    def _denoise_image(self):
        """
        Performs denoising of the image that is currently loaded.
        Updates the image displays with the new output images.
        """
        denoised_image, details_image = denoise(
            self.model, self.noisy_image, self.model_mode == constants.ModelCategory.GSC
        )

        self.denoised_image = denoised_image
        self.details_image = details_image

        self.label_img_denoised.setPixmap(pil_to_qt(denoised_image))
        self.label_mpl_summary.setPixmap(pil_to_qt(details_image))

    def _export_image(self, image: Image.Image, name: str = "Untitled") -> None:
        """
        Exports an image using a file browser.
        :param image: The image to export.
        :param name: The name of the file (without an extension)
        """
        image = image.convert("RGB")
        file_dialog = QFileDialog(self)
        default_name = f"{name}.jpg"
        filters = ";;".join(["Images (*.jpg *.jpeg)", "Images (*.png)", "Images (*.bmp)"])
        path_str, _ = file_dialog.getSaveFileName(self, "Save Image", default_name, filters)

        if path_str:
            path = Path(path_str)
            try:
                image.save(path)
            except Exception as e:
                show_error(text="Something went wrong trying to save your image!")
                print(e)

    def _on_combobox_model_group_current_index_changed(self, idx):
        self.mode = constants.ModelCategory.list()[idx]
        self._update_models()
        self._update_form()

    def _on_list_models_current_item_changed(self, ob):
        iw: ModelListEntry = self.list_models.itemWidget(ob)
        try:
            self.selected_model_path = iw.model_path
        except AttributeError:
            self.selected_model_path = ""
        self._update_form()

    def _on_btn_browse_clicked(self, args):
        self._open_file()
        self._update_form()

    def _on_btn_load_clicked(self, args):
        self._load_model()
        self._update_form()

    def _on_btn_start_clicked(self, args):
        self._denoise_image()
        self._update_form()

    def _on_btn_export_denoised_clicked(self, args):
        self._export_image(self.denoised_image)

    def _on_btn_export_details_clicked(self, args):
        self._export_image(self.details_image)

    def _update_form(self) -> None:
        """
        Updates a form. To be called whenever something changes or happens that could influence
        the availability of some functions.
        """
        # Deps
        ...
        # Conds
        is_model_selected = len(self.selected_model_path) > 0
        is_image_selected = len(self.selected_image_path) > 0
        is_model_loaded = self.model is not None
        is_image_loaded = self.noisy_image is not None and isinstance(self.noisy_image, Image.Image)
        is_image_denoised = self.denoised_image is not None and isinstance(self.denoised_image, Image.Image)
        is_summary_image_loaded = self.details_image is not None and isinstance(self.details_image, Image.Image)

        # Updates
        self.btn_load.setEnabled(is_model_selected)
        self.btn_start.setEnabled(is_model_loaded and is_image_loaded)
        self.btn_export_denoised.setEnabled(is_image_denoised)
        self.btn_export_details.setEnabled(is_summary_image_loaded)

        # Misc
        self._update_status()

def show_error(title: str = "Error", text: str = "An error has occurred.") -> int:
    """
    Displays an error in a user-friendly way.
    :param title: The title of the error
    :param text: The message for the error.
    :return: PySide6 error dialog exit code.
    """

    error_msg = QMessageBox()
    error_msg.setIcon(QMessageBox.Icon.Critical)
    error_msg.setText(text)
    error_msg.setWindowTitle(title)
    error_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    return error_msg.exec()
