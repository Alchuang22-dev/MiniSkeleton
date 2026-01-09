# -*- coding: utf-8 -*-
import sys
from PySide6.QtWidgets import QApplication, QMainWindow

from ui.skeleton_editor import SpotRigWindow
from ui.style import apply_style

class MainWindow(QMainWindow):
    """主窗口：菜单栏、工具栏、停靠面板（骨架编辑/权重/时间线/导出）。"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animation Skeleton Studio")
        # TODO: 初始化 viewport、各 dock 面板并互联

def run():
    app = QApplication(sys.argv)
    apply_style(app)
    win = SpotRigWindow()
    win.show()
    sys.exit(app.exec())
