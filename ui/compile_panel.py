# -*- coding: utf-8 -*-
"""Skeleton compile mode controls."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class SkeletonCompilePanel(QWidget):
    """Controls for compile-mode skeleton editing."""

    def __init__(
        self,
        *,
        on_toggle_mode,
        on_add_joint,
        on_set_parent,
        on_clear_parent,
        on_recompute_weights,
        on_reset_bind,
        on_optimize_quadruped,
        on_export_skeleton,
        on_save_rig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._on_toggle_mode = on_toggle_mode
        self._on_add_joint = on_add_joint
        self._on_set_parent = on_set_parent
        self._on_clear_parent = on_clear_parent
        self._on_recompute_weights = on_recompute_weights
        self._on_reset_bind = on_reset_bind
        self._on_optimize_quadruped = on_optimize_quadruped
        self._on_export_skeleton = on_export_skeleton
        self._on_save_rig = on_save_rig

        self.mode_checkbox: QCheckBox | None = None
        self.parent_combo: QComboBox | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.mode_checkbox = QCheckBox("Skeleton compile mode")
        self.mode_checkbox.toggled.connect(self._on_toggle_mode)
        layout.addWidget(self.mode_checkbox)

        help_label = QLabel(
            "Compile mode edits bind joints without deforming the mesh."
        )
        help_label.setWordWrap(True)
        help_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        help_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(help_label)

        pick_label = QLabel("Tip: Shift-click overlapping joints to cycle selection.")
        pick_label.setWordWrap(True)
        pick_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        pick_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(pick_label)

        self.parent_combo = QComboBox()
        self.parent_combo.addItem("<Root>", -1)
        layout.addWidget(self.parent_combo)

        btn_set_parent = QPushButton("Set parent for selected")
        btn_set_parent.clicked.connect(self._emit_set_parent)
        layout.addWidget(btn_set_parent)

        btn_clear_parent = QPushButton("Clear parent (to root)")
        btn_clear_parent.clicked.connect(self._emit_clear_parent)
        layout.addWidget(btn_clear_parent)

        btn_add_joint = QPushButton("Add joint (child of selected)")
        btn_add_joint.clicked.connect(self._on_add_joint)
        layout.addWidget(btn_add_joint)

        btn_recompute = QPushButton("Recompute weights")
        btn_recompute.clicked.connect(self._on_recompute_weights)
        layout.addWidget(btn_recompute)

        btn_reset_bind = QPushButton("Restore original skeleton")
        btn_reset_bind.clicked.connect(self._on_reset_bind)
        layout.addWidget(btn_reset_bind)

        btn_optimize = QPushButton("Optimize quadruped skeleton")
        btn_optimize.clicked.connect(self._on_optimize_quadruped)
        layout.addWidget(btn_optimize)

        btn_export = QPushButton("Export skeleton (GLB)")
        btn_export.clicked.connect(self._on_export_skeleton)
        layout.addWidget(btn_export)

        btn_save = QPushButton("Save rig as...")
        btn_save.clicked.connect(self._on_save_rig)
        layout.addWidget(btn_save)

        layout.addStretch()

    def _emit_set_parent(self) -> None:
        if self.parent_combo is None:
            return
        parent_idx = self.parent_combo.currentData()
        if parent_idx is None:
            return
        self._on_set_parent(int(parent_idx))

    def _emit_clear_parent(self) -> None:
        if self._on_clear_parent is None:
            return
        self._on_clear_parent()

    def set_joint_names(self, names: list[str]) -> None:
        if self.parent_combo is None:
            return
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        self.parent_combo.addItem("<Root>", -1)
        for idx, name in enumerate(names):
            self.parent_combo.addItem(f"[{idx}] {name}", idx)
        self.parent_combo.blockSignals(False)

    def set_compile_mode(self, enabled: bool) -> None:
        if self.mode_checkbox is None:
            return
        self.mode_checkbox.blockSignals(True)
        self.mode_checkbox.setChecked(bool(enabled))
        self.mode_checkbox.blockSignals(False)
