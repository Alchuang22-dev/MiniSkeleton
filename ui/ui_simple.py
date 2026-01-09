# -*- coding: utf-8 -*-
"""Legacy entrypoint. Prefer ui.skeleton_editor for the split UI."""

from __future__ import annotations

from ui.skeleton_editor import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
