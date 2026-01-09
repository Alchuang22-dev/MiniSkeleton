# -*- coding: utf-8 -*-
"""Background workers for deformation and weight computation."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

import numpy as np

from rigging.lbs import linear_blend_skinning, make_topk_weights
from rigging.weights_heat import HeatWeightsConfig, compute_heat_weights
from ui.weight_tools import compute_simple_weights


class DeformWorker(QObject):
    """Compute deformed vertices off the UI thread."""

    result_ready = Signal(object, int)
    failed = Signal(str, int)

    @Slot(object)
    def compute(self, job: dict) -> None:
        version = int(job.get("version", 0))
        try:
            if job.get("compile_mode", False):
                self.result_ready.emit(job["vertices"], version)
                return

            vertices = job["vertices"]
            skeleton = job["skeleton"]
            weights = job["weights"]
            pose = job["pose"]

            if weights is None:
                self.result_ready.emit(vertices, version)
                return

            skin_mats = skeleton.skinning_matrices(pose)
            deformed = linear_blend_skinning(vertices, weights, skin_mats, topk=None, normalize=False)
            self.result_ready.emit(deformed, version)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc), version)


class WeightsWorker(QObject):
    """Compute heat/simple weights off the UI thread."""

    result_ready = Signal(object, object, object, object)
    failed = Signal(str)

    @Slot(object)
    def compute(self, job: dict) -> None:
        try:
            mesh = job["mesh"]
            skeleton = job["skeleton"]
            cfg: HeatWeightsConfig = job["cfg"]

            weights = compute_heat_weights(mesh, skeleton, cfg)
            topk = min(int(cfg.topk or 4), skeleton.n)
            weights_topk = make_topk_weights(weights, topk)

            bind_locals = [j.bind_local for j in skeleton.joints]
            G_bind = skeleton.forward_kinematics_local(bind_locals)
            joint_positions_fk = G_bind[:, :3, 3]
            simple_weights = compute_simple_weights(mesh.vertices, joint_positions_fk)
            simple_weights_topk = make_topk_weights(simple_weights, 1)

            self.result_ready.emit(weights, weights_topk, simple_weights, simple_weights_topk)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
