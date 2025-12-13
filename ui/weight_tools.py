"""Weight related helper utilities for the rigging UI."""

import numpy as np


def compute_simple_weights(vertices, joint_positions):
    """
    Simplified weights: each vertex only follows its nearest joint (1-hot).
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    joint_positions = np.asarray(joint_positions, dtype=np.float32)

    n_verts = vertices.shape[0]
    n_joints = joint_positions.shape[0]

    distances = np.linalg.norm(
        vertices[:, None, :] - joint_positions[None, :, :],
        axis=2,
    )  # (N,J)

    nearest_joint = np.argmin(distances, axis=1)

    weights = np.zeros((n_verts, n_joints), dtype=np.float32)
    weights[np.arange(n_verts), nearest_joint] = 1.0

    return weights
