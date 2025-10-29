# -*- coding: utf-8 -*-
"""
Heat-kernel skinning weights (Pinocchio-style).

References
----------
Baran & Popović. "Pinocchio: Automatic Rigging and Animation of 3D Characters".
ACM TOG 2007.

Core idea (high level)
----------------------
1) 以“骨段”（parent-child 关节连线）为源（seed），而非单个关节点：
   - 对每个网格顶点 v，计算 v 到每条骨段的最近距离 d(v, bone).
   - 对属于某个关节 j 的“相邻骨段集合”B(j)，取 d_j(v)=min_{bone∈B(j)} d(v,bone)。
   - 用热核（高斯）把距离映射为初始场 s_j(v)=exp( - (d_j / σ)^2 )。

2) 在网格上做“热扩散”/拉普拉斯正则化：
   - 令 L 为（统一）拉普拉斯，构造 A = I - τ * (D^{-1} L)（随机游走标准化）。
   - 对每个关节 j，解 (I - τL_rw) w_j = s_j，得到光滑、空间一致的权重场。

3) 非负 & 归一化 & 稀疏化：
   - 截断负值，行归一化为分割一致性（partition of unity）。
   - 可选：Top-K 裁剪为稀疏权重并重归一化（加速后续 LBS）。

This module is rig/mesh agnostic. It only assumes:
- `mesh`: rigging.mesh_io.Mesh
- `skel`: rigging.skeleton.Skeleton (we access bind_local via FK to get joint positions)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    sp = None
    spla = None

from .mesh_io import Mesh, compute_vertex_adjacency


# -----------------------
# Utilities (bone graph & distances)
# -----------------------

def _global_bind_positions(skel) -> np.ndarray:
    """GLOBAL bind joint positions (J,3)."""
    bind_locals = [j.bind_local for j in skel.joints]
    G = skel.forward_kinematics_local(bind_locals)  # (J,4,4)
    return G[:, :3, 3].astype(np.float32)

def _bone_edges_from_parents(parents: np.ndarray) -> np.ndarray:
    """(B,2) array of (parent, child) indices."""
    return np.array([(p, i) for i, p in enumerate(parents) if p >= 0], dtype=np.int32)

def _incident_bones_per_joint(J: int, edges: np.ndarray) -> List[List[int]]:
    """List of incident bone indices for each joint."""
    inc = [[] for _ in range(J)]
    for b, (p, c) in enumerate(edges):
        inc[p].append(b)
        inc[c].append(b)
    return inc

def _project_point_to_segments_batch(
    V: np.ndarray,  # (Nv,3)
    A: np.ndarray,  # (B,3) bone start (parent)
    B: np.ndarray,  # (B,3) bone end (child)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    t_clamped : (Nv,B) in [0,1]
    dists     : (Nv,B) Euclidean distance from v to closest point on segment
    """
    AB = (B - A).astype(np.float32)            # (B,3)
    ab2 = np.sum(AB * AB, axis=1)              # (B,)
    ab2 = np.maximum(ab2, 1e-20).astype(np.float32)

    Vexp = V[:, None, :]                       # (Nv,1,3)
    Aexp = A[None, :, :]                       # (1,B,3)
    VA = (Vexp - Aexp).astype(np.float32)      # (Nv,B,3)

    t = np.sum(VA * AB[None, :, :], axis=2) / ab2[None, :]  # (Nv,B)
    t_clamped = np.clip(t, 0.0, 1.0)
    C = Aexp + t_clamped[..., None] * AB[None, :, :]        # (Nv,B,3)
    d = np.linalg.norm(Vexp - C, axis=2)                    # (Nv,B)
    return t_clamped.astype(np.float32), d.astype(np.float32)


# -----------------------
# Config
# -----------------------

@dataclass
class HeatWeightsConfig:
    # Heat kernel / diffusion
    sigma: Optional[float] = None        # 距离高斯核 σ；None=>自动（与骨段长度相关）
    tau: float = 0.5                     # 扩散时间步（I - τ L_rw）的 τ
    # Seeds & building
    chunk_size: int = 131072             # 顶点分块大小
    # Post-processing
    topk: Optional[int] = 4              # 每顶点保留的最大影响关节数；None/0 表示不过滤
    smooth_passes: int = 0               # 结果微调平滑次数（拉普拉斯一阶迭代）
    eps: float = 1e-12                   # 数值稳定项
    # Solver
    solver: str = "splu"                 # "splu"（推荐）|"cg"（大网格可用）|"auto"
    cg_tol: float = 1e-6                 # CG 容差
    cg_maxiter: int = 200


# -----------------------
# Main API
# -----------------------

def compute_heat_weights(mesh: Mesh, skel, cfg: Optional[HeatWeightsConfig] = None) -> np.ndarray:
    """
    Pinocchio-style heat weights.

    Parameters
    ----------
    mesh : Mesh
        输入网格（顶点/面）；需要能够生成拉普拉斯（见 mesh_io.compute_vertex_adjacency）
    skel : Skeleton
        骨架（需可从 bind_local 做 FK 获得关节全局位置）
    cfg : HeatWeightsConfig
        超参数（σ、τ、Top-K、solver 等）

    Returns
    -------
    W : (N,J) float32
        每顶点对每关节的权重，非负、行归一化；可在 LBS 中直接使用。
    """
    if cfg is None:
        cfg = HeatWeightsConfig()

    if sp is None or spla is None:
        raise RuntimeError("scipy.sparse is required for heat weights. Please install SciPy.")

    V = np.asarray(mesh.vertices, dtype=np.float32)
    N = V.shape[0]

    parents = skel.parents()
    edges = _bone_edges_from_parents(parents)  # (B,2)
    if edges.size == 0:
        raise ValueError("Skeleton has no bones (no parent-child pairs).")

    J = len(skel.joints)
    Jpos = _global_bind_positions(skel)        # (J,3)
    A = Jpos[edges[:, 0]]                      # (B,3)
    B = Jpos[edges[:, 1]]                      # (B,3)
    seg_len = np.linalg.norm(B - A, axis=1)    # (B,)
    # σ 自动估计：中位骨段长度的一半（经验值）
    if cfg.sigma is None:
        med = float(np.median(seg_len))
        cfg_sigma = max(0.5 * med, 1e-3)
    else:
        cfg_sigma = float(cfg.sigma)

    # 计算网格邻接与拉普拉斯
    neighbors, L = compute_vertex_adjacency(mesh)
    if L is None:
        raise RuntimeError("No Laplacian available (SciPy missing?).")

    # 随机游走标准化 Laplacian：L_rw = D^{-1} L
    deg = np.asarray(L.sum(axis=1)).ravel()
    deg[deg < 1e-16] = 1.0
    Dinv = sp.diags(1.0 / deg, 0, shape=L.shape)
    L_rw = Dinv @ L

    # A = I - tau * L_rw
    I = sp.identity(L.shape[0], format="csr", dtype=np.float32)
    A_sys = (I - cfg.tau * L_rw).tocsr()

    # 预处理/分解（一次，多 RHS 复用）
    solver_mode = cfg.solver.lower()
    do_splu = (solver_mode == "splu" or solver_mode == "auto")
    lu = None
    if do_splu:
        try:
            lu = spla.splu(A_sys.tocsc())
        except Exception:
            # 回退到 CG
            lu = None
            solver_mode = "cg"

    # 为每个关节构造 seed：对其相邻骨段取最小距离 → 高斯核
    inc_bones = _incident_bones_per_joint(J, edges)

    # 为了不爆内存，先整体算顶点到所有骨段的距离（分块），缓存 d_all: (N,B)
    Bcnt = edges.shape[0]
    d_all = np.empty((N, Bcnt), dtype=np.float32)

    bs = int(cfg.chunk_size)
    for s in range(0, N, bs):
        e = min(N, s + bs)
        _t, _d = _project_point_to_segments_batch(V[s:e], A, B)
        d_all[s:e] = _d

    # 构造所有关节的 RHS：s_j = exp(-(min_d / σ)^2)
    S = np.zeros((N, J), dtype=np.float32)
    for j in range(J):
        bones_j = inc_bones[j]
        if len(bones_j) == 0:
            # 没有相邻骨段（例如单根，极少见）：退化为点源（到关节点距离）
            dv = np.linalg.norm(V - Jpos[j][None, :], axis=1).astype(np.float32)
            Sj = np.exp(-(dv / cfg_sigma) ** 2, dtype=np.float32)
        else:
            dj = np.min(d_all[:, bones_j], axis=1)
            Sj = np.exp(-(dj / cfg_sigma) ** 2, dtype=np.float32)
        S[:, j] = Sj

    # 逐关节求解 (I - τ L_rw) w_j = s_j
    W = np.zeros_like(S, dtype=np.float32)

    if lu is not None:
        # 直接 LU 解多个 RHS（按列解）
        for j in range(J):
            rhs = S[:, j].astype(np.float32)
            sol = lu.solve(rhs.astype(np.float64)).astype(np.float32)  # higher precision solve
            W[:, j] = np.maximum(sol, 0.0)
    else:
        # CG 逐列
        for j in range(J):
            rhs = S[:, j].astype(np.float32)
            sol, info = spla.cg(A_sys, rhs, tol=cfg.cg_tol, maxiter=cfg.cg_maxiter)
            if info != 0:
                # 失败回退：用 RHS 作为近似，避免中断
                sol = rhs
            W[:, j] = np.maximum(sol.astype(np.float32), 0.0)

    # 可选：微调平滑（小步数的拉普拉斯一阶迭代），避免噪点
    for _ in range(int(max(0, cfg.smooth_passes))):
        # W <- (I - α L_rw) W, 取 α 为 0.1 * τ 的小步
        alpha = 0.1 * cfg.tau
        W = (W - alpha * (Dinv @ (L @ W))).astype(np.float32)
        W = np.maximum(W, 0.0)

    # Top-K 稀疏化（按行保留最大的 K 个关节）
    if cfg.topk and cfg.topk > 0 and cfg.topk < J:
        k = int(cfg.topk)
        idx = np.argpartition(-W, kth=k-1, axis=1)
        keep = idx[:, :k]
        mask = np.zeros_like(W, dtype=bool)
        rows = np.arange(N)[:, None]
        mask[rows, keep] = True
        W[~mask] = 0.0

    # 行归一化为分割一致性
    rowsum = np.sum(W, axis=1, keepdims=True)
    rowsum[rowsum < cfg.eps] = 1.0
    W /= rowsum

    return W.astype(np.float32)
