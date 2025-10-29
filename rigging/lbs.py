# -*- coding: utf-8 -*-
"""
Linear Blend Skinning (LBS) — model/rig agnostic.

Interfaces
----------
linear_blend_skinning(verts, weights, joint_mats, ...)
    Core LBS for vertices (and optional normals).
    Supports dense np.ndarray (N,J) or scipy.sparse.csr_matrix weights.

Notes
-----
- joint_mats should be "skinning matrices":
    M_skin[j] = G_current[j] @ inv_bind[j]
  You can get them from Skeleton.skinning_matrices(...).
- This implementation is general: it works for biped, quadruped (e.g. cow, dog),
  or arbitrary rigs. The 'quadruped' adaptation lives in skeleton/weights, not here.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def _ensure_homogeneous(verts: np.ndarray, dtype=np.float32) -> np.ndarray:
    """(N,3) -> (N,4) with w=1."""
    v = np.asarray(verts, dtype=dtype)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("verts must be (N,3)")
    ones = np.ones((v.shape[0], 1), dtype=dtype)
    return np.concatenate([v, ones], axis=1)


def _blend_matrices_dense(weights: np.ndarray, joint_mats: np.ndarray) -> np.ndarray:
    """
    Dense weights path.
    weights: (N,J), joint_mats: (J,4,4) -> out: (N,4,4)
    """
    if weights.ndim != 2:
        raise ValueError("weights must be (N,J)")
    if joint_mats.ndim != 3 or joint_mats.shape[1:] != (4, 4):
        raise ValueError("joint_mats must be (J,4,4)")
    # tensordot over J -> (N,4,4)
    return np.tensordot(weights, joint_mats, axes=([1], [0]))


def _blend_matrices_sparse(weights_csr, joint_mats: np.ndarray) -> np.ndarray:
    """
    Sparse weights path (CSR).
    Accumulates per-row M = sum_j w_ij * M_j.
    """
    if sp is None or not sp.isspmatrix_csr(weights_csr):
        raise ValueError("weights must be a scipy.sparse.csr_matrix when using sparse path")
    J = joint_mats.shape[0]
    if joint_mats.ndim != 3 or joint_mats.shape[1:] != (4, 4):
        raise ValueError("joint_mats must be (J,4,4)")
    if weights_csr.shape[1] != J:
        raise ValueError("weights.shape[1] must equal number of joints")

    N = weights_csr.shape[0]
    out = np.zeros((N, 4, 4), dtype=joint_mats.dtype)
    indptr = weights_csr.indptr
    indices = weights_csr.indices
    data = weights_csr.data
    for i in range(N):
        start, end = indptr[i], indptr[i + 1]
        cols = indices[start:end]
        vals = data[start:end]
        if cols.size == 0:
            # No influences: leave identity to avoid NaNs.
            out[i] = np.eye(4, dtype=joint_mats.dtype)
            continue
        # Accumulate
        Mi = np.zeros((4, 4), dtype=joint_mats.dtype)
        for c, w in zip(cols, vals):
            Mi += w * joint_mats[c]
        out[i] = Mi
    return out


def _normalize_weights_inplace(weights: np.ndarray, eps: float = 1e-12) -> None:
    """Row-normalize dense weights in-place."""
    row_sum = np.sum(weights, axis=1, keepdims=True)
    row_sum[row_sum < eps] = 1.0
    weights /= row_sum


def _topk_prune_dense(weights: np.ndarray, k: int) -> None:
    """
    Keep only top-k influences per vertex (dense array version).
    The rest are zeroed; renormalize after pruning.
    """
    if k <= 0 or k >= weights.shape[1]:
        return
    # argsort descending along J, keep top k
    idx = np.argpartition(-weights, kth=k-1, axis=1)
    mask = np.zeros_like(weights, dtype=bool)
    rows = np.arange(weights.shape[0])[:, None]
    topk_cols = idx[:, :k]
    mask[rows, topk_cols] = True
    weights[~mask] = 0.0


def linear_blend_skinning(
    verts: np.ndarray,
    weights,
    joint_mats: np.ndarray,
    *,
    normals: Optional[np.ndarray] = None,
    topk: Optional[int] = None,
    normalize: bool = True,
    return_normals: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    1.4: LBS skinning.

    Parameters
    ----------
    verts : (N,3) float
        Vertices in bind pose.
    weights : (N,J) ndarray or csr_matrix
        Skin weights for each vertex/joint.
    joint_mats : (J,4,4) float
        Skinning matrices (G_curr @ inv_bind) for each joint.
    normals : (N,3) float, optional
        Per-vertex normals (bind pose). If provided and return_normals=True,
        will be transformed using blended linear parts (approximation).
    topk : int, optional
        Keep only top-k influences per vertex (prune the rest). Only for dense weights.
    normalize : bool, default True
        Row-normalize weights (after pruning when dense).
    return_normals : bool, default False
        If True and normals provided, also return transformed normals.

    Returns
    -------
    deformed_verts : (N,3)
    (optional) deformed_normals : (N,3)
    """
    if joint_mats.ndim != 3 or joint_mats.shape[1:] != (4, 4):
        raise ValueError("joint_mats must be (J,4,4)")

    # Prepare homogeneous vertices
    v_h = _ensure_homogeneous(verts, dtype=joint_mats.dtype)  # (N,4)

    # Dense vs sparse handling
    if sp is not None and sp.isspmatrix(weights):
        if not sp.isspmatrix_csr(weights):
            weights = weights.tocsr(copy=True)
        # Sparse path: assume weights already normalized by caller
        M_blend = _blend_matrices_sparse(weights, joint_mats)  # (N,4,4)
    else:
        W = np.asarray(weights, dtype=joint_mats.dtype)
        if W.ndim != 2:
            raise ValueError("weights must be (N,J) dense array or CSR sparse")
        if topk is not None:
            _topk_prune_dense(W, int(topk))
        if normalize:
            _normalize_weights_inplace(W)
        M_blend = _blend_matrices_dense(W, joint_mats)

    # Apply blended matrices
    v_out_h = (M_blend @ v_h[..., None]).squeeze(-1)  # (N,4)
    v_out = v_out_h[:, :3] / np.clip(v_out_h[:, 3:4], 1e-12, None)

    if not return_normals or normals is None:
        return v_out

    # --- normals (approximate): transform by linear part (no translation), then renormalize
    n = np.asarray(normals, dtype=joint_mats.dtype)
    if n.ndim != 2 or n.shape[1] != 3 or n.shape[0] != verts.shape[0]:
        raise ValueError("normals must be (N,3) and match verts shape")

    # linear part of M_blend: (N,3,3)
    R_blend = M_blend[:, :3, :3]
    n_out = (R_blend @ n[..., None]).squeeze(-1)
    # normalize normals
    n_norm = np.linalg.norm(n_out, axis=1, keepdims=True)
    n_norm[n_norm < 1e-20] = 1.0
    n_out /= n_norm

    return v_out, n_out
