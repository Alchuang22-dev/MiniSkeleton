# -*- coding: utf-8 -*-
"""Shared Qt stylesheet and setup helpers."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication


APP_STYLE = """
QWidget {
    background-color: #f5f5f5;
    color: #222222;
    font-size: 12px;
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
    color: #333333;
}
QComboBox::drop-down {
    border: none;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #666666;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: white;
    color: #333333;
    selection-background-color: #4CAF50;
    selection-color: white;
    border: 1px solid #cccccc;
}
QLabel#TitleLabel {
    font-size: 16px;
    font-weight: bold;
    color: #333333;
}
QLabel#FooterLabel {
    font-size: 10px;
    color: #999999;
}
"""


def apply_style(app: QApplication) -> None:
    """Apply a consistent style across the UI."""
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
