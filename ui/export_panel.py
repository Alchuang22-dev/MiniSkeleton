"""Left side control/toolbar panel."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RigControlPanel(QWidget):
    """Contains reset/skinning controls and embeds the timeline widget."""

    def __init__(self, *, timeline_widget: QWidget, on_reset, on_skinning_change, parent=None):
        super().__init__(parent)
        self.timeline_widget = timeline_widget
        self.on_reset = on_reset
        self.on_skinning_change = on_skinning_change

        self.skinning_combo: QComboBox | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.setFixedWidth(260)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #000000;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#resetButton {
                background-color: #ff9800;
            }
            QPushButton#resetButton:hover {
                background-color: #e68900;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
                color: #333;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #666;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #333;
                selection-background-color: #4CAF50;
                selection-color: white;
                border: 1px solid #cccccc;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Spot Rig Tool")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # Controls
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        reset_button = QPushButton("Reset to bind pose")
        reset_button.setObjectName("resetButton")
        reset_button.clicked.connect(self.on_reset)
        control_layout.addWidget(reset_button)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Skinning
        skinning_group = QGroupBox("Skinning")
        skinning_layout = QVBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: normal; color: #555;")
        skinning_layout.addWidget(mode_label)

        self.skinning_combo = QComboBox()
        self.skinning_combo.addItem("Full skinning (heat weights)", "full")
        self.skinning_combo.addItem("Nearest-joint 1-hot", "simple")
        self.skinning_combo.currentIndexChanged.connect(self._emit_skinning_change)
        skinning_layout.addWidget(self.skinning_combo)

        mode_info = QLabel(
            "- Full: smooth heat-kernel weights (Pinocchio style)\n"
            "- Simple: each vertex follows its nearest joint"
        )
        mode_info.setStyleSheet(
            "font-size: 11px; color: #555; "
            "background-color: #fff; padding: 8px; "
            "border-radius: 3px; border: 1px solid #ddd;"
        )
        mode_info.setWordWrap(True)
        skinning_layout.addWidget(mode_info)

        skinning_group.setLayout(skinning_layout)
        layout.addWidget(skinning_group)

        # Timeline
        anim_group = QGroupBox("Animation / Keyframes")
        anim_layout = QVBoxLayout()
        anim_layout.addWidget(self.timeline_widget)
        anim_group.setLayout(anim_layout)
        layout.addWidget(anim_group)

        layout.addStretch()

        info_label = QLabel("Spot Demo · Heat Weights + LBS + Keyframes")
        info_label.setStyleSheet("font-size: 10px; color: #999;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

    # ------------------------------------------------------------------ API

    def _emit_skinning_change(self, index: int):
        if self.skinning_combo is None:
            return
        mode = self.skinning_combo.itemData(index)
        self.on_skinning_change(mode)

    def set_skinning_mode(self, mode: str):
        if self.skinning_combo is None:
            return
        idx = self.skinning_combo.findData(mode)
        if idx >= 0:
            self.skinning_combo.setCurrentIndex(idx)
