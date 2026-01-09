"""Left side control/toolbar panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.compile_panel import SkeletonCompilePanel


class RigControlPanel(QWidget):
    """Contains reset/skinning controls and embeds the timeline widget."""

    def __init__(
        self,
        *,
        timeline_widget: QWidget,
        on_reset,
        on_skinning_change,
        on_toggle_compile=None,
        on_add_joint=None,
        on_set_parent=None,
        on_recompute_weights=None,
        on_save_rig=None,
        parent=None,
    ):
        super().__init__(parent)
        self.timeline_widget = timeline_widget
        self.on_reset = on_reset
        self.on_skinning_change = on_skinning_change

        self.skinning_combo: QComboBox | None = None
        self.compile_panel: SkeletonCompilePanel | None = None

        self._compile_callbacks = {
            "on_toggle_compile": on_toggle_compile,
            "on_add_joint": on_add_joint,
            "on_set_parent": on_set_parent,
            "on_recompute_weights": on_recompute_weights,
            "on_save_rig": on_save_rig,
        }
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.setFixedWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Spot Rig Tool")
        title.setObjectName("TitleLabel")
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

        # Skeleton compile controls (optional)
        if self._compile_callbacks["on_toggle_compile"] is not None:
            compile_group = QGroupBox("Skeleton Compile")
            compile_layout = QVBoxLayout()
            self.compile_panel = SkeletonCompilePanel(
                on_toggle_mode=self._compile_callbacks["on_toggle_compile"],
                on_add_joint=self._compile_callbacks["on_add_joint"],
                on_set_parent=self._compile_callbacks["on_set_parent"],
                on_recompute_weights=self._compile_callbacks["on_recompute_weights"],
                on_save_rig=self._compile_callbacks["on_save_rig"],
            )
            compile_layout.addWidget(self.compile_panel)
            compile_group.setLayout(compile_layout)
            layout.addWidget(compile_group)

        # Timeline
        anim_group = QGroupBox("Animation / Keyframes")
        anim_layout = QVBoxLayout()
        anim_layout.addWidget(self.timeline_widget)
        anim_group.setLayout(anim_layout)
        layout.addWidget(anim_group)

        layout.addStretch()

        info_label = QLabel("Spot Demo · Heat Weights + LBS + Keyframes")
        info_label.setObjectName("FooterLabel")
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

    def set_joint_names(self, names: list[str]) -> None:
        if self.compile_panel is None:
            return
        self.compile_panel.set_joint_names(names)

    def set_compile_mode(self, enabled: bool) -> None:
        if self.compile_panel is None:
            return
        self.compile_panel.set_compile_mode(enabled)
