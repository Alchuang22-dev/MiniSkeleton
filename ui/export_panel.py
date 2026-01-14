"""Left side control/toolbar panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
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
        on_clear_parent=None,
        on_recompute_weights=None,
        on_reset_bind=None,
        on_optimize_quadruped=None,
        on_export_skeleton=None,
        on_render_frames=None,
        on_make_video=None,
        on_save_rig=None,
        on_export_joint_positions=None,
        on_export_bone_edges=None,
        on_export_weights=None,
        parent=None,
    ):
        super().__init__(parent)
        self.timeline_widget = timeline_widget
        self.on_reset = on_reset
        self.on_skinning_change = on_skinning_change
        self.on_load_model = on_load_model
        self.on_render_frames = on_render_frames
        self.on_make_video = on_make_video
        self.on_export_joint_positions = on_export_joint_positions
        self.on_export_bone_edges = on_export_bone_edges
        self.on_export_weights = on_export_weights

        self.skinning_combo: QComboBox | None = None
        self.compile_panel: SkeletonCompilePanel | None = None
        self.model_path_edit: QLineEdit | None = None
        self.frames_path_edit: QLineEdit | None = None
        self.video_path_edit: QLineEdit | None = None
        self.fps_spin: QSpinBox | None = None
        self.width_spin: QSpinBox | None = None
        self.height_spin: QSpinBox | None = None

        self._compile_callbacks = {
            "on_toggle_compile": on_toggle_compile,
            "on_add_joint": on_add_joint,
            "on_set_parent": on_set_parent,
            "on_clear_parent": on_clear_parent,
            "on_recompute_weights": on_recompute_weights,
            "on_reset_bind": on_reset_bind,
            "on_optimize_quadruped": on_optimize_quadruped,
            "on_export_skeleton": on_export_skeleton,
            "on_save_rig": on_save_rig,
        }
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.setMinimumWidth(520)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(10)

        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(10)
        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        right_column = QVBoxLayout()
        right_column.setSpacing(10)

        def add_group(widget: QWidget, column: int) -> None:
            if column == 0:
                left_column.addWidget(widget)
            else:
                right_column.addWidget(widget)

        title = QLabel("Spot Rig Tool")
        title.setObjectName("TitleLabel")
        outer_layout.addWidget(title)

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
            add_group(model_group, 0)

        # Controls
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        reset_button = QPushButton("Reset to bind pose")
        reset_button.setObjectName("resetButton")
        reset_button.clicked.connect(self.on_reset)
        control_layout.addWidget(reset_button)
        control_group.setLayout(control_layout)
        add_group(control_group, 0)

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
        add_group(skinning_group, 0)

        # Skeleton compile controls (optional)
        if self._compile_callbacks["on_toggle_compile"] is not None:
            compile_group = QGroupBox("Skeleton Compile")
            compile_layout = QVBoxLayout()
            self.compile_panel = SkeletonCompilePanel(
                on_toggle_mode=self._compile_callbacks["on_toggle_compile"],
                on_add_joint=self._compile_callbacks["on_add_joint"],
                on_set_parent=self._compile_callbacks["on_set_parent"],
                on_clear_parent=self._compile_callbacks["on_clear_parent"],
                on_recompute_weights=self._compile_callbacks["on_recompute_weights"],
                on_reset_bind=self._compile_callbacks["on_reset_bind"],
                on_optimize_quadruped=self._compile_callbacks["on_optimize_quadruped"],
                on_export_skeleton=self._compile_callbacks["on_export_skeleton"],
                on_save_rig=self._compile_callbacks["on_save_rig"],
            )
            compile_layout.addWidget(self.compile_panel)
            compile_group.setLayout(compile_layout)
            add_group(compile_group, 1)

        # Video export
        if self.on_render_frames is not None:
            video_group = QGroupBox("Video Export")
            video_layout = QVBoxLayout()

            self.frames_path_edit = QLineEdit()
            self.frames_path_edit.setPlaceholderText("Frames dir (e.g., out/frames/spot)")
            video_layout.addWidget(self.frames_path_edit)

            fps_row = QHBoxLayout()
            fps_label = QLabel("FPS")
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 120)
            self.fps_spin.setValue(30)
            fps_row.addWidget(fps_label)
            fps_row.addWidget(self.fps_spin)
            video_layout.addLayout(fps_row)

            size_row = QHBoxLayout()
            self.width_spin = QSpinBox()
            self.width_spin.setRange(128, 4096)
            self.width_spin.setValue(1024)
            self.height_spin = QSpinBox()
            self.height_spin.setRange(128, 4096)
            self.height_spin.setValue(1024)
            size_row.addWidget(QLabel("W"))
            size_row.addWidget(self.width_spin)
            size_row.addWidget(QLabel("H"))
            size_row.addWidget(self.height_spin)
            video_layout.addLayout(size_row)

            btn_render = QPushButton("Render frames (VTK)")
            btn_render.clicked.connect(self._emit_render_frames)
            video_layout.addWidget(btn_render)

            self.video_path_edit = QLineEdit()
            self.video_path_edit.setPlaceholderText("Video path (e.g., out/videos/spot.mp4)")
            video_layout.addWidget(self.video_path_edit)

            btn_video = QPushButton("Make video")
            btn_video.clicked.connect(self._emit_make_video)
            video_layout.addWidget(btn_video)

            video_group.setLayout(video_layout)
            add_group(video_group, 1)

        # Debug export
        if (
            self.on_export_joint_positions is not None
            or self.on_export_bone_edges is not None
            or self.on_export_weights is not None
        ):
            debug_group = QGroupBox("Debug Export")
            debug_layout = QVBoxLayout()

            if self.on_export_joint_positions is not None:
                btn_joints = QPushButton("Export joint positions")
                btn_joints.clicked.connect(self.on_export_joint_positions)
                debug_layout.addWidget(btn_joints)

            if self.on_export_bone_edges is not None:
                btn_edges = QPushButton("Export skeleton edges")
                btn_edges.clicked.connect(self.on_export_bone_edges)
                debug_layout.addWidget(btn_edges)

            if self.on_export_weights is not None:
                btn_weights = QPushButton("Export bind weights")
                btn_weights.clicked.connect(self.on_export_weights)
                debug_layout.addWidget(btn_weights)

            debug_group.setLayout(debug_layout)
            add_group(debug_group, 1)

        # Timeline
        anim_group = QGroupBox("Animation / Keyframes")
        anim_layout = QVBoxLayout()
        anim_layout.addWidget(self.timeline_widget)
        anim_group.setLayout(anim_layout)
        add_group(anim_group, 0)
        columns_layout.addLayout(left_column, 1)
        columns_layout.addLayout(right_column, 1)
        outer_layout.addLayout(columns_layout)

        info_label = QLabel("Spot Demo · Heat Weights + LBS + Keyframes")
        info_label.setObjectName("FooterLabel")
        info_label.setAlignment(Qt.AlignCenter)
        outer_layout.addWidget(info_label)
        outer_layout.addStretch(1)

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

    def _emit_render_frames(self) -> None:
        if self.on_render_frames is None or self.frames_path_edit is None:
            return
        out_dir = self.frames_path_edit.text().strip()
        if not out_dir:
            return
        fps = self.fps_spin.value() if self.fps_spin is not None else 30
        width = self.width_spin.value() if self.width_spin is not None else 1024
        height = self.height_spin.value() if self.height_spin is not None else 1024
        self.on_render_frames(out_dir, fps, width, height)

    def _emit_make_video(self) -> None:
        if self.on_make_video is None or self.frames_path_edit is None or self.video_path_edit is None:
            return
        frames_dir = self.frames_path_edit.text().strip()
        video_path = self.video_path_edit.text().strip()
        if not frames_dir or not video_path:
            return
        fps = self.fps_spin.value() if self.fps_spin is not None else 30
        width = self.width_spin.value() if self.width_spin is not None else 1024
        height = self.height_spin.value() if self.height_spin is not None else 1024
        self.on_make_video(frames_dir, video_path, fps, width, height)

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

    def set_video_paths(self, frames_dir: str, video_path: str) -> None:
        if self.frames_path_edit is not None:
            self.frames_path_edit.setText(frames_dir)
        if self.video_path_edit is not None:
            self.video_path_edit.setText(video_path)

    def set_joint_names(self, names: list[str]) -> None:
        if self.compile_panel is None:
            return
        self.compile_panel.set_joint_names(names)

    def set_compile_mode(self, enabled: bool) -> None:
        if self.compile_panel is None:
            return
        self.compile_panel.set_compile_mode(enabled)
