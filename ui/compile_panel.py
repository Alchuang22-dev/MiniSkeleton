# -*- coding: utf-8 -*-
"""Skeleton compile mode controls."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
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
        on_recompute_weights,
        on_save_rig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._on_toggle_mode = on_toggle_mode
        self._on_add_joint = on_add_joint
        self._on_set_parent = on_set_parent
        self._on_recompute_weights = on_recompute_weights
        self._on_save_rig = on_save_rig

        self.mode_checkbox: QCheckBox | None = None
        self.parent_combo: QComboBox | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.mode_checkbox = QCheckBox("Skeleton compile mode")
        self.mode_checkbox.toggled.connect(self._on_toggle_mode)
        layout.addWidget(self.mode_checkbox)

        help_label = QLabel(
            "Compile mode edits bind joints without deforming the mesh."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.parent_combo = QComboBox()
        self.parent_combo.addItem("<Root>", -1)
        layout.addWidget(self.parent_combo)

        btn_set_parent = QPushButton("Set parent for selected")
        btn_set_parent.clicked.connect(self._emit_set_parent)
        layout.addWidget(btn_set_parent)

        btn_add_joint = QPushButton("Add joint (child of selected)")
        btn_add_joint.clicked.connect(self._on_add_joint)
        layout.addWidget(btn_add_joint)

        btn_recompute = QPushButton("Recompute weights")
        btn_recompute.clicked.connect(self._on_recompute_weights)
        layout.addWidget(btn_recompute)

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
