# rigging/gltf_loader.py
# -*- coding: utf-8 -*-
"""
Minimal glTF/GLB mesh + skeleton loader for Spot.

功能：
- 从 .glb/.gltf 中读取第一个 skin 的关节列表
- 解析 joints 的名字、父子关系、bind pose 下的关节位置
- 从同一个 skin 绑定的 mesh 里读取顶点和三角面
- 对骨架和网格统一做同一个坐标系旋转（绕 X 轴 -90°），方便和现有 UI/算法对齐
"""

from __future__ import annotations
from typing import List, Tuple
import base64
import numpy as np
from pygltflib import GLTF2


# ------------------ buffer / accessor helpers ------------------ #

def _load_buffer_bytes(gltf: GLTF2, buffer_index: int) -> bytes:
    buf = gltf.buffers[buffer_index]
    if buf.uri is None:
        # binary glb
        return gltf.binary_blob()
    if buf.uri.startswith("data:"):
        header, encoded = buf.uri.split(",", 1)
        return base64.b64decode(encoded)
    raise RuntimeError("External buffer file not supported in this loader")


def _load_accessor(gltf: GLTF2, accessor_id: int) -> np.ndarray:
    """
    统一处理：
    - FLOAT / UNSIGNED_SHORT / UNSIGNED_INT / UNSIGNED_BYTE
    - SCALAR / VEC3 / VEC4 / MAT4
    并正确考虑 accessor.byteOffset 与 bufferView.byteOffset。
    """
    acc = gltf.accessors[accessor_id]
    bv = gltf.bufferViews[acc.bufferView]

    # 组件类型 -> dtype
    ct = acc.componentType
    if ct == 5126:      # FLOAT
        dt = np.float32
    elif ct == 5123:    # UNSIGNED_SHORT
        dt = np.uint16
    elif ct == 5125:    # UNSIGNED_INT
        dt = np.uint32
    elif ct == 5121:    # UNSIGNED_BYTE
        dt = np.uint8
    else:
        raise NotImplementedError(f"Unsupported componentType {ct}")

    # 每个元素包含的组件个数
    comp_per_elem = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }.get(acc.type)
    if comp_per_elem is None:
        raise NotImplementedError(f"Unsupported accessor type {acc.type}")

    # 原始 buffer
    raw_buf = _load_buffer_bytes(gltf, bv.buffer)

    # 总偏移 = bufferView.byteOffset + accessor.byteOffset
    base_offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)

    count = acc.count * comp_per_elem
    elem_size = np.dtype(dt).itemsize
    byte_len = count * elem_size

    raw_slice = raw_buf[base_offset: base_offset + byte_len]
    arr = np.frombuffer(raw_slice, dtype=dt)

    # 整理形状
    if acc.type == "SCALAR":
        # indices: 统一转为 int32
        return arr.astype(np.int32)
    elif acc.type == "VEC3":
        return arr.reshape(-1, 3)
    elif acc.type == "VEC4":
        return arr.reshape(-1, 4)
    elif acc.type == "MAT4":
        return arr.reshape(-1, 16)
    else:
        # 其他类型暂不需要
        raise NotImplementedError(f"Unsupported accessor type {acc.type}")


# ------------------ transform helpers ------------------ #

def _node_local_matrix(node) -> np.ndarray:
    """把 glTF 的 node (matrix / TRS) 转成 4x4 局部矩阵."""
    M = np.eye(4, dtype=np.float32)

    if node.matrix:
        M[:] = np.array(node.matrix, dtype=np.float32).reshape(4, 4)
        return M

    # TRS 形式
    t = np.array(node.translation or [0.0, 0.0, 0.0], dtype=np.float32)
    r = np.array(node.rotation    or [0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # x,y,z,w
    s = np.array(node.scale       or [1.0, 1.0, 1.0], dtype=np.float32)

    # 四元数 -> 旋转矩阵
    x, y, z, w = r
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       1 - 2*(xx+zz),     2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx+yy)]
    ], dtype=np.float32)

    R = R * s[None, :]
    M[:3, :3] = R
    M[:3, 3]  = t
    return M


def _compute_node_globals(gltf: GLTF2) -> np.ndarray:
    """为场景中所有 node 计算全局 4x4 矩阵 (world transform)。"""
    N = len(gltf.nodes)
    local = np.zeros((N, 4, 4), dtype=np.float32)
    parents = np.full(N, -1, dtype=np.int32)

    # local transforms
    for i, node in enumerate(gltf.nodes):
        local[i] = _node_local_matrix(node)

    # parent 索引：根据 children 反推
    for pid, parent in enumerate(gltf.nodes):
        if not parent.children:
            continue
        for cid in parent.children:
            parents[cid] = pid

    global_mats = np.zeros_like(local)
    visited = np.zeros(N, dtype=bool)

    def dfs(i):
        if visited[i]:
            return
        p = parents[i]
        if p == -1:
            global_mats[i] = local[i]
        else:
            dfs(p)
            global_mats[i] = global_mats[p] @ local[i]
        visited[i] = True

    for i in range(N):
        dfs(i)

    return global_mats


# 统一坐标系旋转（绕 X 轴 -90°）：x'=x, y'=-z, z'=y
_ROT_X_NEG_90 = np.array(
    [
        [1.0,  0.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  1.0,  0.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ],
    dtype=np.float32,
)


# ------------------ public loaders ------------------ #

def load_mesh_and_skeleton_from_glb(path: str):
    """
    从 GLB 读取 mesh + skeleton（优先使用 IBM，异常时自动回退到 node global）。
    返回:
        vertices : (N,3) float32  世界坐标
        faces    : (F,3) int32
        names    : list[str]
        parents  : (J,) int32
        positions: (J,3) float32  关节 bind pose 世界坐标
    """
    print(f"\n==================== [GLB] load_mesh_and_skeleton_from_glb ====================")
    print(f"📦 读取 GLB: {path}")

    gltf = GLTF2().load(path)

    if not gltf.skins:
        raise RuntimeError(f"GLB '{path}' has no skins (skeleton)")

    skin_index = 0
    skin = gltf.skins[skin_index]
    joint_nodes = skin.joints
    J = len(joint_nodes)
    print(f"  ▶ skins: {len(gltf.skins)}, 使用 skin[{skin_index}]，关节数 J={J}")

    # ---------- 1) 所有 node 的 global 矩阵 ----------
    node_globals = _compute_node_globals(gltf)  # (N,4,4)

    # ---------- 2) mesh_nodes: 收集所有使用该 skin 的 mesh node ----------
    mesh_nodes: List[int] = []
    for nid, node in enumerate(gltf.nodes):
        if node.mesh is not None and node.skin == skin_index:
            mesh_nodes.append(nid)

    if not mesh_nodes:
        # 没有显式绑定 skin，就退回到第一个带 mesh 的 node
        for nid, node in enumerate(gltf.nodes):
            if node.mesh is not None:
                mesh_nodes.append(nid)
                break

    if not mesh_nodes:
        raise RuntimeError(f"GLB '{path}' has no mesh node")

    print(f"  ▶ mesh_nodes (使用的 node id): {mesh_nodes}")

    # ---------- 3) 合并所有 mesh primitive，计算 mesh AABB ----------
    vertices_list = []
    faces_list = []
    vert_offset = 0

    for mesh_node_index in mesh_nodes:
        mesh_node = gltf.nodes[mesh_node_index]
        mesh_idx = mesh_node.mesh
        mesh_def = gltf.meshes[mesh_idx]
        if not mesh_def.primitives:
            continue

        M_mesh = node_globals[mesh_node_index]

        for prim in mesh_def.primitives:
            attrs = prim.attributes
            pos_accessor_index = getattr(attrs, "POSITION", None)
            if pos_accessor_index is None:
                continue

            pos_local = _load_accessor(gltf, pos_accessor_index).astype(np.float32)  # (n,3)
            homo = np.concatenate(
                [pos_local, np.ones((pos_local.shape[0], 1), dtype=np.float32)],
                axis=1,
            )  # (n,4)
            pos_world = (M_mesh @ homo.T).T[:, :3]  # (n,3)

            if prim.indices is not None:
                faces_flat = _load_accessor(gltf, prim.indices).astype(np.int32)
                faces = faces_flat.reshape(-1, 3) + vert_offset
            else:
                n = pos_world.shape[0]
                faces = (np.arange(n, dtype=np.int32).reshape(-1, 3) + vert_offset)

            vertices_list.append(pos_world)
            faces_list.append(faces)
            vert_offset += pos_world.shape[0]

    if not vertices_list:
        raise RuntimeError("No POSITION data found for any mesh primitive")

    vertices = np.concatenate(vertices_list, axis=0).astype(np.float32)
    faces = np.concatenate(faces_list, axis=0).astype(np.int32)

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    vcenter = (vmin + vmax) * 0.5
    scale = np.linalg.norm(vmax - vmin)
    print(f"  ▶ glb vertices: {vertices.shape}")
    print(f"  ▶ glb faces   : {faces.shape}")
    print(f"  ▶ mesh AABB min: {vmin}")
    print(f"  ▶ mesh AABB max: {vmax}")
    print(f"  ▶ mesh center  : {vcenter}")
    print(f"  ▶ mesh scale   : {scale}")

    # ---------- 4) skeleton: 名字 + 父子关系 ----------
    names: List[str] = []
    for nid in joint_nodes:
        node = gltf.nodes[nid]
        names.append(node.name if node.name else f"joint_{nid}")

    node_to_joint = {nid: j for j, nid in enumerate(joint_nodes)}
    parents = np.full(J, -1, dtype=np.int32)

    for parent_node_id, parent_node in enumerate(gltf.nodes):
        if not parent_node.children:
            continue
        for child_id in parent_node.children:
            if child_id in node_to_joint and parent_node_id in node_to_joint:
                c = node_to_joint[child_id]
                p = node_to_joint[parent_node_id]
                parents[c] = p

    print(f"  ▶ parents (前10): {parents[:10]}")

    # ---------- 5) 两套候选的 joint global 矩阵：A=IBM, B=node_globals ----------
    # 选 mesh_nodes[0] 做绑定参考
    ref_mesh_node = mesh_nodes[0]
    T_mesh = node_globals[ref_mesh_node]  # (4,4)
    print(f"  ▶ T_mesh (ref node={ref_mesh_node}) 平移: {T_mesh[:3, 3]}")

    # B: 直接 node_globals
    G_B = np.zeros((J, 4, 4), dtype=np.float32)
    for j, nid in enumerate(joint_nodes):
        G_B[j] = node_globals[nid]
    pos_B = G_B[:, :3, 3]
    center_B = pos_B.mean(axis=0)
    print(f"  ▶ candidate B (node_globals) joint center: {center_B}")
    print(f"    B - mesh center: {center_B - vcenter}")

    # A: 通过 IBM 反推
    use_IBM = skin.inverseBindMatrices is not None
    G_A = None
    pos_A = None
    center_A = None

    if use_IBM:
        ibm_flat = _load_accessor(gltf, skin.inverseBindMatrices).astype(np.float32)  # (J,16)
        ibm = ibm_flat.reshape(-1, 4, 4)  # 先按 accessor 原样 reshape

        print(f"  ▶ inverseBindMatrices shape: {ibm.shape}")
        print(f"    IBM[0] raw:\n{ibm[0]}")

        # ⚠ glTF 存的是列主，pygltflib 直接给出来的通常是“转置版”，
        # 可以看到平移在最后一行，所以这里要统一成我们自己用的行主格式：
        # [ R | t ]
        # [ 0 | 1 ]
        ibm = np.transpose(ibm, (0, 2, 1))  # (J,4,4) 逐个转置
        print(f"    IBM[0] transposed:\n{ibm[0]}")

        G_A = np.zeros((J, 4, 4), dtype=np.float32)
        for j in range(J):
            # 正确公式：G_bind_j = T_mesh * inv(IBM_j)
            G_A[j] = T_mesh @ np.linalg.inv(ibm[j])

        pos_A = G_A[:, :3, 3]
        center_A = pos_A.mean(axis=0)
        print(f"  ▶ candidate A (IBM) joint center: {center_A}")
        print(f"    A - mesh center: {center_A - vcenter}")

    # ---------- 6) 选哪一套关节矩阵？ ----------
    if use_IBM and center_A is not None:
        dist_A = np.linalg.norm(center_A - vcenter)
        dist_B = np.linalg.norm(center_B - vcenter)
        print(f"  ▶ dist_A(IBM)={dist_A:.6f}, dist_B(nodes)={dist_B:.6f}")

        if np.isfinite(dist_A) and dist_A < 3.0 * dist_B:
            print("  ✅ 使用 IBM 推导的关节矩阵 (candidate A)")
            joint_globals = G_A
        else:
            print("  ⚠️ IBM 推导结果看起来异常，回退到 node_globals (candidate B)")
            joint_globals = G_B
    else:
        print("  ▶ 没有 IBM 或 IBM 读取失败，使用 node_globals (candidate B)")
        joint_globals = G_B

    positions = joint_globals[:, :3, 3].astype(np.float32)
    center_final = positions.mean(axis=0)
    print(f"  ▶ 最终 joint center: {center_final}")
    print(f"    final - mesh center: {center_final - vcenter}")

    return vertices, faces, names, parents, positions


def load_skeleton_from_glb(path: str):
    """
    兼容旧接口：只要骨架信息。
    """
    _, _, names, parents, positions = load_mesh_and_skeleton_from_glb(path)
    return names, parents, positions
