# -*- coding: utf-8 -*-
"""
Nearest-bilinear skinning weights (model/rig agnostic).

Idea
----
For each vertex:
  1) Find the K=2 closest bones (segments defined by parent-child joints).
  2) For each bone, project vertex to the segment -> get axial t in [0,1]
     and radial distance d.
  3) Axial linear blend: [ (1-t) for parent_joint,  t for child_joint ].
  4) Radial weight falloff (Gaussian by default) to mix the two nearest bones.
     => "bi-linear": along-bone (axis) linear + across-bone linear.
  5) Accumulate contributions into a (N,J) dense matrix; keep up to 4 influences
     per vertex (2 bones × 2 endpoints). Normalize row-wise.

This is a robust generalization of "nearest bone / dual bone interpolation"
that works well on quadrupeds or arbitrary rigs without any cow-specific code.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp  # optional: we output dense here, but can convert if needed
except Exception:
    sp = None

# -----------------------
# Helpers to read skeleton bind positions
# -----------------------

def _global_bind_positions(skel) -> np.ndarray:
    """
    Get GLOBAL bind joint positions (J,3) from Skeleton.
    Works with the provided rigging/skeleton.py implementation.
    """
    # G_bind = FK of bind_local only
    bind_locals = [j.bind_local for j in skel.joints]
    G_bind = skel.forward_kinematics_local(bind_locals)  # (J,4,4)
    return G_bind[:, :3, 3].astype(np.float32)


def _bone_edges_from_parents(parents: np.ndarray) -> np.ndarray:
    """Return (B,2) int array of (parent, child) joint index pairs."""
    edges = np.array(
        [(p, i) for i, p in enumerate(parents) if p >= 0],
        dtype=np.int32
    )
    return edges


# -----------------------
# Core: nearest-bilinear weights
# -----------------------

@dataclass
class NearestBilinearConfig:
    k_bones: int = 2            # number of nearest bones
    max_influences: int = 4     # per-vertex max non-zeros (2 bones × 2 endpoints)
    sigma: Optional[float] = None  # radial falloff; if None, auto from bone lengths
    chunk_size: int = 131072    # process vertices by chunks; tune for memory
    falloff: str = "gaussian"   # or "inv_dist"
    eps: float = 1e-12
    renormalize: bool = True


def _auto_sigma(seg_lengths: np.ndarray) -> float:
    """
    Heuristic sigma: median segment length * 0.5 (works well empirically).
    """
    med = float(np.median(seg_lengths))
    if med <= 0.0 or not np.isfinite(med):
        return 1.0
    return max(med * 0.5, 1e-3)


def _radial_weight(d: np.ndarray, sigma: float, mode: str = "gaussian", eps: float = 1e-12) -> np.ndarray:
    if mode == "gaussian":
        # exp(-d^2 / (2*sigma^2))
        s2 = 2.0 * (sigma ** 2)
        return np.exp(- (d * d) / max(s2, 1e-12)).astype(np.float32)
    elif mode == "inv_dist":
        return (1.0 / np.maximum(d, eps)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported falloff mode: {mode}")


def _project_points_to_segments_batch(
    V: np.ndarray,          # (Nv,3) chunk of vertices
    A: np.ndarray,          # (B,3) bone start
    B: np.ndarray,          # (B,3) bone end
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project each vertex to each segment.

    Returns
    -------
    t_clamped : (Nv,B) in [0,1]
    dists     : (Nv,B) Euclidean distance from v to closest point on segment
    """
    # segment vectors
    AB = (B - A).astype(np.float32)             # (B,3)
    ab2 = np.sum(AB * AB, axis=1)               # (B,)
    ab2 = np.maximum(ab2, 1e-20).astype(np.float32)

    # V to A vectors per bone: we compute v - A for all v; do expansion based on (Nv,1,3) - (1,B,3)
    V_exp = V[:, None, :]                        # (Nv,1,3)
    A_exp = A[None, :, :]                        # (1,B,3)
    VA = (V_exp - A_exp).astype(np.float32)      # (Nv,B,3)

    # t = dot(VA, AB) / ||AB||^2
    t = np.sum(VA * AB[None, :, :], axis=2) / ab2[None, :]  # (Nv,B)
    t_clamped = np.clip(t, 0.0, 1.0)

    # closest point C = A + t*AB
    C = A_exp + t_clamped[..., None] * AB[None, :, :]       # (Nv,B,3)
    d = np.linalg.norm(V_exp - C, axis=2)                   # (Nv,B)

    return t_clamped.astype(np.float32), d.astype(np.float32), ab2.astype(np.float32)


def compute_nearest_bilinear_weights(
    verts: np.ndarray,            # (N,3)
    skel,                         # rigging.skeleton.Skeleton
    config: Optional[NearestBilinearConfig] = None,
) -> np.ndarray:
    """
    Compute dense weights (N,J) using nearest-bilinear rule.

    Steps per vertex:
      - pick K=2 nearest bones by distance to segment,
      - axial weights on each bone: [(1-t), t],
      - multiply by radial falloff per bone,
      - accumulate into up to 4 joints (two endpoints per bone),
      - prune to max_influences and renormalize.

    Returns
    -------
    W : (N,J) float32
    """
    if config is None:
        config = NearestBilinearConfig()

    V = np.asarray(verts, dtype=np.float32)
    N = V.shape[0]
    parents = skel.parents()
    edges = _bone_edges_from_parents(parents)    # (B,2)
    if edges.size == 0:
        raise ValueError("Skeleton has no bones (no parent-child pairs).")

    # Bind joint positions and bone endpoints
    J_pos = _global_bind_positions(skel)        # (J,3)
    A = J_pos[edges[:, 0]]                      # (B,3) parent
    B = J_pos[edges[:, 1]]                      # (B,3) child
    AB = B - A
    seg_lengths = np.linalg.norm(AB, axis=1)    # (B,)

    sigma = config.sigma if (config.sigma is not None) else _auto_sigma(seg_lengths)

    # Prepare output
    J = J_pos.shape[0]
    W = np.zeros((N, J), dtype=np.float32)

    # Chunked processing to limit memory footprint
    bs = int(config.chunk_size)
    for start in range(0, N, bs):
        end = min(N, start + bs)
        Vc = V[start:end]  # (Nv,3)

        t, d, _ab2 = _project_points_to_segments_batch(Vc, A, B)  # (Nv,B) each

        # pick k nearest bones per vertex
        K = int(max(1, config.k_bones))
        if K >= t.shape[1]:
            K = t.shape[1]

        # indices of K smallest distances along axis=1
        # argpartition faster than argsort
        idx_part = np.argpartition(d, K-1, axis=1)[:, :K]           # (Nv,K)
        # sort those K by distance ascending for stability
        row_ids = np.arange(idx_part.shape[0])[:, None]
        d_sorted = np.take_along_axis(d, idx_part, axis=1)
        order = np.argsort(d_sorted, axis=1)
        bone_ids = np.take_along_axis(idx_part, order, axis=1)      # (Nv,K)
        d_sorted = np.take_along_axis(d_sorted, order, axis=1)
        t_sorted = np.take_along_axis(t, idx_part, axis=1)
        t_sorted = np.take_along_axis(t_sorted, order, axis=1)

        # radial falloff per bone
        F = _radial_weight(d_sorted, sigma=sigma, mode=config.falloff, eps=config.eps)  # (Nv,K)

        # accumulate contributions to joints
        for k in range(K):
            b_ids = bone_ids[:, k]                   # (Nv,)
            t_k = t_sorted[:, k]                     # (Nv,)
            f_k = F[:, k]                            # (Nv,)

            j_parent = edges[b_ids, 0]               # (Nv,)
            j_child  = edges[b_ids, 1]               # (Nv,)

            w_parent = (1.0 - t_k) * f_k
            w_child  = t_k * f_k

            # scatter-add
            W[start:end, j_parent] += w_parent
            W[start:end, j_child]  += w_child

        # prune to max_influences if requested (dense path)
        if config.max_influences and config.max_influences > 0:
            mi = int(config.max_influences)
            # keep top-mi per row
            # argpartition descending
            # NOTE: we operate on the chunk only
            idx_desc = np.argpartition(-W[start:end], kth=mi-1, axis=1)
            keep_cols = idx_desc[:, :mi]
            mask = np.zeros_like(W[start:end], dtype=bool)
            rr = np.arange(end - start)[:, None]
            mask[rr, keep_cols] = True
            W[start:end][~mask] = 0.0

        # renormalize chunk
        if config.renormalize:
            rowsum = np.sum(W[start:end], axis=1, keepdims=True)
            rowsum[rowsum < config.eps] = 1.0
            W[start:end] /= rowsum

    return W
