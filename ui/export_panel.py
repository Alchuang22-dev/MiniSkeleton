"""Left side control/toolbar panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
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
        on_load_model=None,
        on_toggle_compile=None,
        on_add_joint=None,
        on_set_parent=None,
        on_recompute_weights=None,
        on_reset_bind=None,
        on_optimize_quadruped=None,
        on_export_skeleton=None,
        on_save_rig=None,
        parent=None,
    ):
        super().__init__(parent)
        self.timeline_widget = timeline_widget
        self.on_reset = on_reset
        self.on_skinning_change = on_skinning_change
        self.on_load_model = on_load_model

        self.skinning_combo: QComboBox | None = None
        self.compile_panel: SkeletonCompilePanel | None = None
        self.model_path_edit: QLineEdit | None = None

        self._compile_callbacks = {
            "on_toggle_compile": on_toggle_compile,
            "on_add_joint": on_add_joint,
            "on_set_parent": on_set_parent,
            "on_recompute_weights": on_recompute_weights,
            "on_reset_bind": on_reset_bind,
            "on_optimize_quadruped": on_optimize_quadruped,
            "on_export_skeleton": on_export_skeleton,
            "on_save_rig": on_save_rig,
        }
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.setFixedWidth(260)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Spot Rig Tool")
        title.setObjectName("TitleLabel")
        layout.addWidget(title)

        # Model file
        if self.on_load_model is not None:
            model_group = QGroupBox("Model")
            model_layout = QVBoxLayout()

            path_row = QHBoxLayout()
            self.model_path_edit = QLineEdit()
            self.model_path_edit.setPlaceholderText("Path to .glb")
            path_row.addWidget(self.model_path_edit)

            btn_browse = QPushButton("Browse")
            btn_browse.clicked.connect(self._browse_model)
            path_row.addWidget(btn_browse)
            model_layout.addLayout(path_row)

            btn_load = QPushButton("Load")
            btn_load.clicked.connect(self._emit_load_model)
            model_layout.addWidget(btn_load)

            model_group.setLayout(model_layout)
            layout.addWidget(model_group)

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
                on_reset_bind=self._compile_callbacks["on_reset_bind"],
                on_optimize_quadruped=self._compile_callbacks["on_optimize_quadruped"],
                on_export_skeleton=self._compile_callbacks["on_export_skeleton"],
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

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

    # ------------------------------------------------------------------ API

    def _emit_skinning_change(self, index: int):
        if self.skinning_combo is None:
            return
        mode = self.skinning_combo.itemData(index)
        self.on_skinning_change(mode)

    def _browse_model(self) -> None:
        if self.model_path_edit is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model file",
            "",
            "GLB Files (*.glb);;All Files (*)",
        )
        if not path:
            return
        self.model_path_edit.setText(path)

    def _emit_load_model(self) -> None:
        if self.on_load_model is None or self.model_path_edit is None:
            return
        path = self.model_path_edit.text().strip()
        if not path:
            return
        self.on_load_model(path)

    def set_skinning_mode(self, mode: str):
        if self.skinning_combo is None:
            return
        idx = self.skinning_combo.findData(mode)
        if idx >= 0:
            self.skinning_combo.setCurrentIndex(idx)

    def set_model_path(self, path: str) -> None:
        if self.model_path_edit is None:
            return
        self.model_path_edit.setText(path)

    def set_joint_names(self, names: list[str]) -> None:
        if self.compile_panel is None:
            return
        self.compile_panel.set_joint_names(names)

    def set_compile_mode(self, enabled: bool) -> None:
        if self.compile_panel is None:
            return
        self.compile_panel.set_compile_mode(enabled)
