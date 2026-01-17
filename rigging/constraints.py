"""Minimal IK/constraint utilities (not yet wired into UI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class IKResult:
    positions: np.ndarray
    reached: bool
    iterations: int


@dataclass
class IKChain:
    indices: List[int]

    @staticmethod
    def from_end(parents: Sequence[int], end_idx: int, length: int) -> "IKChain":
        indices = [int(end_idx)]
        cur = int(end_idx)
        for _ in range(max(0, length - 1)):
            if cur < 0:
                break
            cur = int(parents[cur])
            if cur < 0:
                break
            indices.append(cur)
        indices.reverse()
        return IKChain(indices)


def solve_fabrik(
    positions: np.ndarray,
    chain: Iterable[int],
    target: Sequence[float],
    *,
    max_iters: int = 15,
    tolerance: float = 1e-3,
) -> IKResult:
    pos = np.asarray(positions, dtype=np.float32).copy()
    chain_idx = [int(i) for i in chain]
    if len(chain_idx) < 2:
        return IKResult(pos, True, 0)

    target = np.asarray(target, dtype=np.float32)
    lengths = []
    for i in range(len(chain_idx) - 1):
        a = pos[chain_idx[i]]
        b = pos[chain_idx[i + 1]]
        lengths.append(float(np.linalg.norm(b - a)))

    root_idx = chain_idx[0]
    root_pos = pos[root_idx].copy()
    total_len = float(np.sum(lengths))
    if total_len <= 1e-8:
        return IKResult(pos, True, 0)

    if float(np.linalg.norm(target - root_pos)) >= total_len:
        for i in range(len(chain_idx) - 1):
            cur = chain_idx[i]
            nxt = chain_idx[i + 1]
            r = float(np.linalg.norm(target - pos[cur]))
            if r <= 1e-8:
                continue
            lam = lengths[i] / r
            pos[nxt] = (1.0 - lam) * pos[cur] + lam * target
        return IKResult(pos, False, 1)

    reached = False
    iterations = 0
    for it in range(max_iters):
        iterations = it + 1
        pos[chain_idx[-1]] = target
        for i in range(len(chain_idx) - 2, -1, -1):
            cur = chain_idx[i]
            nxt = chain_idx[i + 1]
            r = float(np.linalg.norm(pos[nxt] - pos[cur]))
            if r <= 1e-8:
                continue
            lam = lengths[i] / r
            pos[cur] = (1.0 - lam) * pos[nxt] + lam * pos[cur]

        pos[root_idx] = root_pos
        for i in range(len(chain_idx) - 1):
            cur = chain_idx[i]
            nxt = chain_idx[i + 1]
            r = float(np.linalg.norm(pos[nxt] - pos[cur]))
            if r <= 1e-8:
                continue
            lam = lengths[i] / r
            pos[nxt] = (1.0 - lam) * pos[cur] + lam * pos[nxt]

        if float(np.linalg.norm(pos[chain_idx[-1]] - target)) <= tolerance:
            reached = True
            break

    return IKResult(pos, reached, iterations)
