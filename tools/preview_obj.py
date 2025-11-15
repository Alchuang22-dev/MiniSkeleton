# -*- coding: utf-8 -*-
"""
tools/preview_obj.py

用途：
- 快速预览 OBJ/PLY 等网格文件；
- 调用 rigging.mesh_io.Mesh 做读写与拓扑检查；
- 可选地用 PyVista 打开一个简单的 3D 预览窗口。
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

try:
    import pyvista as pv  # type: ignore
except Exception:  # noqa: BLE001
    pv = None

from rigging.mesh_io import Mesh


def summarize_mesh(mesh: Mesh) -> None:
    """打印网格的一些基础统计信息。"""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)

    print("=== Mesh Summary ===")
    print(f"  ▶ vertices: {v.shape}")
    print(f"  ▶ faces   : {f.shape}")
    if v.size > 0:
        aabb_min = v.min(axis=0)
        aabb_max = v.max(axis=0)
        diag = np.linalg.norm(aabb_max - aabb_min)
        print(f"  ▶ AABB min: {aabb_min}")
        print(f"  ▶ AABB max: {aabb_max}")
        print(f"  ▶ diagonal length: {diag:.6f}")
    if hasattr(mesh, "check_topology"):
        try:
            print("\n=== Topology Check ===")
            topo_info = mesh.check_topology()
            # topo_info 可以是字符串 / dict，这里做一个宽松打印
            print(topo_info)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 拓扑检查失败: {exc}")


def preview_mesh(mesh: Mesh, title: Optional[str] = None) -> None:
    """使用 PyVista 进行简单预览。"""
    if pv is None:
        print("[WARN] 未安装 pyvista，无法进行 3D 预览。可使用:")
        print("       pip install pyvista pyvistaqt")
        return

    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)

    # faces 为 (F, 3) -> PyVista 需要 [3, i0, i1, i2] 形式
    faces_with_count = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f])

    plotter = pv.Plotter()
    plotter.set_background("white")

    mesh_pv = pv.PolyData(v, faces_with_count)
    plotter.add_mesh(
        mesh_pv,
        color="lightblue",
        opacity=0.9,
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        smooth_shading=True,
    )
    plotter.add_axes()
    plotter.show_grid(color="lightgray")
    if title:
        plotter.add_text(title, font_size=12)

    plotter.show()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="预览 OBJ/PLY 网格并进行简单拓扑检查")
    parser.add_argument("path", help="要加载的 OBJ/PLY/GLB 等网格文件路径")
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="只打印网格信息，不打开 3D 预览窗口",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.path):
        raise SystemExit(f"[ERROR] 文件不存在: {args.path}")

    print(f"📦 加载网格: {args.path}")
    # 兼容两种 Mesh API：Mesh.load / Mesh.from_file
    mesh = (
        Mesh.load(args.path)
        if hasattr(Mesh, "load")
        else Mesh.from_file(args.path)  # type: ignore[attr-defined]
    )

    # 可选：补法线
    if hasattr(mesh, "ensure_vertex_normals"):
        mesh.ensure_vertex_normals(recompute=True)

    summarize_mesh(mesh)

    if not args.no_view:
        preview_mesh(mesh, title=os.path.basename(args.path))


if __name__ == "__main__":
    main()
