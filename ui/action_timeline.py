"""Keyframe timeline widget for the rig editor."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class ActionTimeline(QWidget):
    """
    Encapsulates keyframe record/clear/playback logic in a small panel.
    """

    def __init__(
        self,
        *,
        get_transforms,
        set_transforms,
        on_update_mesh,
        on_status,
        parent=None,
    ):
        super().__init__(parent)
        self.get_transforms = get_transforms
        self.set_transforms = set_transforms
        self.on_update_mesh = on_update_mesh
        self.on_status = on_status

        self.keyframes: list[np.ndarray] = []
        self.current_frame_index: int = -1
        self.is_playing: bool = False

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)  # ~30 FPS
        self.play_timer.timeout.connect(self._play_step)

        self.play_button: QPushButton | None = None
        self.status_label: QLabel | None = None

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        layout = QVBoxLayout(self)

        btn_record = QPushButton("Record keyframe")
        btn_record.clicked.connect(self.record_keyframe)
        layout.addWidget(btn_record)

        btn_clear = QPushButton("Clear keyframes")
        btn_clear.clicked.connect(self.clear_keyframes)
        layout.addWidget(btn_clear)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        self.status_label = QLabel("Keyframes: 0 | Current: - | State: stopped")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

    # ----------------------------------------------------------------- Logic

    def record_keyframe(self):
        transforms = self.get_transforms()
        if transforms is None:
            self._notify("No skeleton loaded; cannot record keyframe.")
            return
        self.keyframes.append(np.array(transforms, copy=True))
        self.current_frame_index = len(self.keyframes) - 1
        self._notify(f"Recorded keyframe #{self.current_frame_index}")
        self._refresh_status()

    def clear_keyframes(self):
        self.stop_playback()
        self.keyframes.clear()
        self.current_frame_index = -1
        self._notify("Cleared all keyframes.")
        self._refresh_status()

    def toggle_playback(self):
        if not self.is_playing:
            if not self.keyframes:
                self._notify("No keyframes to play; record at least one.")
                return
            self.start_playback()
        else:
            self.stop_playback()

    def start_playback(self):
        if not self.keyframes:
            return
        self.is_playing = True
        if self.current_frame_index < 0:
            self.current_frame_index = 0
        self.play_timer.start()
        if self.play_button is not None:
            self.play_button.setText("Pause")
        self._refresh_status()

    def stop_playback(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.play_timer.stop()
        if self.play_button is not None:
            self.play_button.setText("Play")
        self._refresh_status()

    def _play_step(self):
        if not self.keyframes:
            self.stop_playback()
            return

        self.current_frame_index = (self.current_frame_index + 1) % len(self.keyframes)
        self.set_transforms(np.array(self.keyframes[self.current_frame_index], copy=True))
        self.on_update_mesh()
        self._refresh_status()

    def reset(self):
        """Reset timeline when skeleton is reloaded."""
        self.stop_playback()
        self.keyframes.clear()
        self.current_frame_index = -1
        self._refresh_status()

    # ---------------------------------------------------------------- Helpers

    def _refresh_status(self):
        if self.status_label is None:
            return
        count = len(self.keyframes)
        state = "playing" if self.is_playing else "stopped"
        idx_str = self.current_frame_index if self.current_frame_index >= 0 else "-"
        self.status_label.setText(f"Keyframes: {count} | Current: {idx_str} | State: {state}")

    def _notify(self, msg: str):
        if self.on_status:
            self.on_status(msg)
