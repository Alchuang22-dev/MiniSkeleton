"""Main rig editor window and controller."""

from __future__ import annotations

import sys
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
)

from rigging.mesh_io import Mesh
from rigging.gltf_loader import load_mesh_and_skeleton_from_glb
from rigging.skeleton import Skeleton
from rigging.lbs import linear_blend_skinning, make_topk_weights
from rigging.weights_heat import HeatWeightsConfig, compute_heat_weights
from rigging.rig_io import save_rig_npz
from rigging.skeleton_optimize import optimize_quadruped_bind_positions
from rigging.gltf_export import export_skeleton_glb as gltf_export_skeleton

from ui.action_timeline import ActionTimeline
from ui.compute_worker import DeformWorker, WeightsWorker
from ui.export_panel import RigControlPanel
from ui.viewport import RigViewport
from ui.weight_tools import compute_simple_weights
from ui.style import apply_style


class SpotRigWindow(QMainWindow):
    """Spot model rig binding demo with a small keyframe timeline."""

    compute_requested = Signal(object)
    weights_requested = Signal(object)

    def __init__(self):
        super().__init__()

        # Core data
        self.mesh: Mesh | None = None
        self.skeleton: Skeleton | None = None
        self.bones: list[tuple[int, int]] = []
        self.joint_names: list[str] = []
        self.joint_parents: list[int] = []
        self.bind_positions: np.ndarray | None = None
        self._orig_joint_names: list[str] = []
        self._orig_joint_parents: list[int] = []
        self._orig_bind_positions: np.ndarray | None = None
        self.weights = None
        self.simple_weights = None
        self.weights_topk = None
        self.simple_weights_topk = None
        self.joint_transforms = None  # (J,4,4)
        self.initial_joint_transforms = None
        self.weights_dirty = False
        self._pending_joint_idx: int | None = None
        self._pending_joint_delta: np.ndarray | None = None
        self._latest_state_version = 0
        self._pending_compute = False
        self._compute_inflight = False
        self._weights_inflight = False

        # Interaction state
        self.selected_joint: int | None = None
        self.skinning_mode = "full"
        self.compile_mode = False

        # UI components
        self.timeline: ActionTimeline | None = None
        self.toolbar: RigControlPanel | None = None
        self.viewport: RigViewport | None = None

        self._tick_timer: QTimer | None = None
        self._deform_thread: QThread | None = None
        self._weights_thread: QThread | None = None

        self._init_workers()
        self._init_ui()
        self.model_path = "data/single/spot/spot.glb"
        self.load_model(self.model_path)
        self._start_tick()

        if self.mesh is not None:
            self.mesh.ensure_vertex_normals(recompute=True)

    # ------------------------------------------------------------------ UI

    def _init_ui(self):
        self.setWindowTitle("Spot Rig Tool (Heat Weights + LBS + Keyframes)")
        self.setGeometry(100, 100, 1400, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.timeline = ActionTimeline(
            get_transforms=self.get_transforms,
            set_transforms=self.set_transforms,
            on_update_mesh=self.request_deform_update,
            on_status=self._notify,
        )
        self.toolbar = RigControlPanel(
            timeline_widget=self.timeline,
            on_reset=self.reset_to_initial,
            on_skinning_change=self.on_skinning_mode_changed,
            on_load_model=self.load_model_from_path,
            on_toggle_compile=self.set_compile_mode,
            on_add_joint=self.add_joint,
            on_set_parent=self.reparent_selected_joint,
            on_recompute_weights=self.recompute_weights,
            on_reset_bind=self.reset_bind_pose,
            on_optimize_quadruped=self.optimize_quadruped_skeleton,
            on_export_skeleton=self.export_skeleton_glb,
            on_save_rig=self.save_rig_bundle,
        )
        self.viewport = RigViewport(self, self._notify)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.addWidget(self.toolbar)
        splitter.addWidget(self.viewport)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1140])
        layout.addWidget(splitter)

        self.statusBar().showMessage("Click red sphere to select a joint; drag arrows to move.")

    # --------------------------------------------------------------- Workers

    def _init_workers(self) -> None:
        self._deform_thread = QThread(self)
        self._deform_worker = DeformWorker()
        self._deform_worker.moveToThread(self._deform_thread)
        self.compute_requested.connect(self._deform_worker.compute)
        self._deform_worker.result_ready.connect(self._on_deform_ready)
        self._deform_worker.failed.connect(self._on_deform_failed)
        self._deform_thread.start()

        self._weights_thread = QThread(self)
        self._weights_worker = WeightsWorker()
        self._weights_worker.moveToThread(self._weights_thread)
        self.weights_requested.connect(self._weights_worker.compute)
        self._weights_worker.result_ready.connect(self._on_weights_ready)
        self._weights_worker.failed.connect(self._on_weights_failed)
        self._weights_thread.start()

    def _start_tick(self) -> None:
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(16)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start()

    # ---------------------------------------------------------------- Loading

    def load_model(self, path: str):
        """Load spot.glb mesh + skeleton and initialize weights/pose."""
        try:
            glb_path = path
            print("\n==================== [STEP 1] LOAD MESH + SKELETON FROM GLB ====================")
            print(f"[INFO] Loading mesh + skeleton from: {glb_path}")

            verts, faces, names, parents, joint_positions = load_mesh_and_skeleton_from_glb(glb_path)
            print(f"  - glb vertices: {verts.shape}")
            print(f"  - glb faces   : {faces.shape}")

            self.mesh = Mesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int32))
            self.mesh.ensure_vertex_normals(recompute=True)

            V = self.mesh.vertices
            mesh_aabb_min = V.min(axis=0)
            mesh_aabb_max = V.max(axis=0)
            mesh_center = (mesh_aabb_min + mesh_aabb_max) * 0.5
            mesh_scale = np.linalg.norm(mesh_aabb_max - mesh_aabb_min)
            print(f"  - mesh AABB min: {mesh_aabb_min}")
            print(f"  - mesh AABB max: {mesh_aabb_max}")
            print(f"  - mesh center  : {mesh_center}")
            print(f"  - mesh scale   : {mesh_scale}")

            print("\n==================== [STEP 2] BUILD SKELETON ====================")
            self.skeleton = Skeleton.from_bind_positions(names, parents, joint_positions)
            print(f"  - Skeleton joints: {self.skeleton.n}")
            self.joint_names = list(names)
            self.joint_parents = [int(p) for p in parents]
            self.bind_positions = np.array(joint_positions, dtype=np.float32, copy=True)
            self._orig_joint_names = list(self.joint_names)
            self._orig_joint_parents = list(self.joint_parents)
            self._orig_bind_positions = np.array(self.bind_positions, copy=True)

            self.bones = [
                (j.parent, i)
                for i, j in enumerate(self.skeleton.joints)
                if j.parent is not None and j.parent >= 0
            ]
            print(f"  - bones (edges): {len(self.bones)}")

            bind_locals = [j.bind_local for j in self.skeleton.joints]
            G_bind = self.skeleton.forward_kinematics_local(bind_locals)
            print(f"  - G_bind shape: {G_bind.shape}")
            print(f"  - G_bind first 5 positions:\n{G_bind[:5, :3, 3]}")

            fk_center = G_bind[:, :3, 3].mean(axis=0)
            print(f"  - FK joint center: {fk_center}")
            print(f"  - FK - mesh center: {fk_center - mesh_center}")

            print("\n==================== [STEP 4] HEAT WEIGHTS ====================")
            cfg = HeatWeightsConfig(tau=0.5, topk=4, smooth_passes=1)
            print("[INFO] Computing heat weights (Pinocchio style)...")
            self.weights = compute_heat_weights(self.mesh, self.skeleton, cfg)
            print("  - Heat weights shape:", self.weights.shape)
            topk = min(int(cfg.topk or 4), self.skeleton.n)
            self.weights_topk = make_topk_weights(self.weights, topk)

            print("\n==================== [STEP 5] SIMPLE WEIGHTS ====================")
            joint_positions_fk = G_bind[:, :3, 3]
            self.simple_weights = compute_simple_weights(self.mesh.vertices, joint_positions_fk)
            print("  - Simple weights computed")
            self.simple_weights_topk = make_topk_weights(self.simple_weights, 1)
            self.weights_dirty = False

            print("\n==================== [STEP 6] INIT TRANSFORMS ====================")
            J = self.skeleton.n
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(J, axis=0)
            self.initial_joint_transforms = self.joint_transforms.copy()
            if self.timeline:
                self.timeline.reset()
            print("  - transforms initialized")

            print("\n==================== [STEP 7] RENDER ====================")
            self.update_viewport_full()
            if self.toolbar:
                self.toolbar.set_joint_names(self.joint_names)
                self.toolbar.set_compile_mode(self.compile_mode)
                self.toolbar.set_model_path(glb_path)
            self._notify(f"Spot(glb) loaded: {self.skeleton.n} joints, {self.mesh.n_vertices} vertices")
        except Exception as e:  # noqa: BLE001
            print("Load failed:", e)
            import traceback

            traceback.print_exc()
            self._notify(f"Load failed: {e}")

    def load_model_from_path(self, path: str) -> None:
        if not path:
            return
        self.model_path = path
        if self.timeline:
            self.timeline.reset()
        self.load_model(path)

    # -------------------------------------------------------------- Callbacks

    def reset_to_initial(self):
        if self.initial_joint_transforms is None:
            self._notify("No initial pose to reset.")
            return
        if self.compile_mode:
            self.reset_bind_pose()
            return
        self.set_transforms(np.array(self.initial_joint_transforms, copy=True))
        self.selected_joint = None
        self._pending_compute = True
        self._notify("Reset to bind pose.")

    def on_skinning_mode_changed(self, mode: str):
        self.skinning_mode = mode
        self._latest_state_version += 1
        self._pending_compute = True
        self._notify(f"Switched skinning mode to: {mode}")

    def set_compile_mode(self, enabled: bool) -> None:
        self.compile_mode = bool(enabled)
        if self.compile_mode:
            if self.timeline:
                self.timeline.stop_playback()
            if self.joint_transforms is not None:
                self.joint_transforms[:] = np.eye(4, dtype=np.float32)
            self._notify("Compile mode enabled: mesh stays static.")
        else:
            if self.weights_dirty:
                self._notify("Compile mode disabled. Weights may be stale; recompute.")
        self._latest_state_version += 1
        if self.toolbar:
            self.toolbar.set_compile_mode(self.compile_mode)
        self._pending_compute = True

    # --------------------------------------------------------------- Geometry

    def compute_deformed_vertices(self):
        if self.compile_mode:
            return self.mesh.vertices
        if self.weights is None or self.simple_weights is None:
            return self.mesh.vertices
        pose = np.array(self.joint_transforms, copy=True)
        M_skin = self.skeleton.skinning_matrices(pose)
        if self.skinning_mode == "simple":
            weights = self.simple_weights_topk or self.simple_weights
        else:
            weights = self.weights_topk or self.weights
        return linear_blend_skinning(
            self.mesh.vertices,
            weights,
            M_skin,
            topk=None,
            normalize=False,
        )

    def compute_current_global_mats(self):
        if self.compile_mode:
            bind_locals = [j.bind_local for j in self.skeleton.joints]
            return self.skeleton.forward_kinematics_local(bind_locals)
        pose = [self.joint_transforms[j] for j in range(self.skeleton.n)]
        return self.skeleton.forward_kinematics_pose(pose)

    def get_joint_children(self, joint_idx):
        children = []
        parents = self.joint_parents or [j.parent for j in self.skeleton.joints]
        for i, parent in enumerate(parents):
            if parent == joint_idx:
                children.append(i)
        return children

    def update_children_cascade(self, parent_idx, delta):
        children = self.get_joint_children(parent_idx)
        for child_idx in children:
            self.joint_transforms[child_idx][:3, 3] += delta
            self.update_children_cascade(child_idx, delta)

    def queue_joint_delta(self, joint_idx: int, delta: np.ndarray) -> None:
        if self._pending_joint_idx is None or self._pending_joint_idx != joint_idx:
            self._pending_joint_idx = joint_idx
            self._pending_joint_delta = np.array(delta, dtype=np.float32)
        else:
            self._pending_joint_delta += delta

    def _translate_bind_positions(self, joint_idx: int, delta: np.ndarray) -> None:
        if self.bind_positions is None:
            return
        indices = self._collect_subtree(joint_idx)
        self.bind_positions[indices] += delta[None, :]
        self._rebuild_skeleton_from_bind()

    def _collect_subtree(self, root_idx: int) -> list[int]:
        out: list[int] = []
        stack = [root_idx]
        while stack:
            idx = stack.pop()
            out.append(idx)
            for child in self.get_joint_children(idx):
                stack.append(child)
        return out

    def _rebuild_skeleton_from_bind(self) -> None:
        if self.bind_positions is None:
            return
        self.skeleton = Skeleton.from_bind_positions(
            self.joint_names,
            self.joint_parents,
            self.bind_positions,
        )
        self.bones = [
            (j.parent, i)
            for i, j in enumerate(self.skeleton.joints)
            if j.parent is not None and j.parent >= 0
        ]
        J = self.skeleton.n
        self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(J, axis=0)
        self.initial_joint_transforms = self.joint_transforms.copy()
        self.weights = None
        self.simple_weights = None
        self.weights_topk = None
        self.simple_weights_topk = None
        self.weights_dirty = True

    # ----------------------------------------------------------- Compile UI

    def add_joint(self) -> None:
        if self.mesh is None or self.bind_positions is None:
            self._notify("No mesh loaded; cannot add joint.")
            return
        mesh_min = self.mesh.vertices.min(axis=0)
        mesh_max = self.mesh.vertices.max(axis=0)
        mesh_center = (mesh_min + mesh_max) * 0.5
        mesh_diag = float(np.linalg.norm(mesh_max - mesh_min))
        offset = np.array([0.0, 0.05 * mesh_diag, 0.0], dtype=np.float32)

        parent_idx = self.selected_joint if self.selected_joint is not None else -1
        if parent_idx >= 0:
            base_pos = self.bind_positions[parent_idx]
        else:
            base_pos = mesh_center

        new_name = self._unique_joint_name("joint")
        new_pos = (base_pos + offset).astype(np.float32)

        self.joint_names.append(new_name)
        self.joint_parents.append(int(parent_idx))
        self.bind_positions = np.vstack([self.bind_positions, new_pos[None, :]])

        self._rebuild_skeleton_from_bind()
        self.selected_joint = len(self.joint_names) - 1
        if self.toolbar:
            self.toolbar.set_joint_names(self.joint_names)
        self.update_viewport_full()
        self._notify(f"Added joint [{self.selected_joint}] {new_name}.")

    def reparent_selected_joint(self, parent_idx: int) -> None:
        if self.selected_joint is None:
            self._notify("Select a joint first.")
            return
        child_idx = self.selected_joint
        if parent_idx == child_idx:
            self._notify("Cannot set a joint as its own parent.")
            return
        if parent_idx >= 0 and self._is_descendant(parent_idx, child_idx):
            self._notify("Invalid parent: would create a cycle.")
            return

        self.joint_parents[child_idx] = int(parent_idx)
        self._rebuild_skeleton_from_bind()
        if self.toolbar:
            self.toolbar.set_joint_names(self.joint_names)
        self.update_viewport_full()
        self._notify(f"Reparented joint [{child_idx}] to {parent_idx}.")

    def reset_bind_pose(self) -> None:
        if self._orig_bind_positions is None:
            self._notify("No original bind pose available.")
            return
        self.joint_names = list(self._orig_joint_names)
        self.joint_parents = list(self._orig_joint_parents)
        self.bind_positions = np.array(self._orig_bind_positions, copy=True)
        self._rebuild_skeleton_from_bind()
        if self.toolbar:
            self.toolbar.set_joint_names(self.joint_names)
        self.recompute_weights()
        self.update_viewport_full()
        self._notify("Bind pose reset to original.")

    def optimize_quadruped_skeleton(self) -> None:
        if self.mesh is None or self.bind_positions is None:
            self._notify("No mesh loaded; cannot optimize.")
            return
        if not self.compile_mode:
            self._notify("Enable compile mode before optimizing skeleton.")
            return
        new_pos, report = optimize_quadruped_bind_positions(
            self.mesh,
            np.asarray(self.joint_parents, dtype=np.int32),
            self.bind_positions,
        )
        self.bind_positions = new_pos
        self._rebuild_skeleton_from_bind()
        if self.toolbar:
            self.toolbar.set_joint_names(self.joint_names)
        self.update_viewport_full()
        msg = f"Optimize quadruped: axis={report.axis}, delta={report.delta:.4f}"
        if report.warnings:
            msg += " | " + "; ".join(report.warnings)
        self._notify(msg)

    def export_skeleton_glb(self) -> None:
        if self.skeleton is None:
            self._notify("No skeleton loaded; cannot export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export skeleton to GLB",
            "out/rigs/skeleton.glb",
            "GLB Files (*.glb)",
        )
        if not path:
            return
        if not path.lower().endswith(".glb"):
            path = f"{path}.glb"
        gltf_export_skeleton(
            self.skeleton,
            path,
            names=self.joint_names,
            parents=self.joint_parents,
        )
        self._notify(f"Skeleton exported: {path}")

    def recompute_weights(self) -> None:
        if self.mesh is None or self.skeleton is None:
            self._notify("No skeleton loaded; cannot compute weights.")
            return
        if self._weights_inflight:
            self._notify("Weights computation already running.")
            return
        cfg = HeatWeightsConfig(tau=0.5, topk=4, smooth_passes=1)
        self._weights_inflight = True
        self._notify("Computing weights...")
        self.weights_requested.emit({"mesh": self.mesh, "skeleton": self.skeleton, "cfg": cfg})

    def save_rig_bundle(self) -> None:
        if self.mesh is None or self.skeleton is None:
            self._notify("No skeleton loaded; cannot save rig.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save rig bundle",
            "out/rigs/rig.npz",
            "Rig Bundle (*.npz)",
        )
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path = f"{path}.npz"
        save_rig_npz(
            path,
            self.mesh,
            self.skeleton,
            self.joint_names,
            self.joint_parents,
        )
        self._notify(f"Saved rig bundle: {path}")

    def _unique_joint_name(self, base: str) -> str:
        existing = set(self.joint_names)
        idx = len(self.joint_names)
        name = f"{base}_{idx}"
        while name in existing:
            idx += 1
            name = f"{base}_{idx}"
        return name

    def _is_descendant(self, node_idx: int, ancestor_idx: int) -> bool:
        p = node_idx
        while p is not None and p >= 0:
            if p == ancestor_idx:
                return True
            p = self.joint_parents[p]
        return False

    # ------------------------------------------------------- Tick & workers

    def _on_tick(self) -> None:
        if self._pending_joint_idx is not None and self._pending_joint_delta is not None:
            idx = self._pending_joint_idx
            delta = self._pending_joint_delta
            self._pending_joint_idx = None
            self._pending_joint_delta = None
            if self.compile_mode:
                self._translate_bind_positions(idx, delta)
            else:
                self.joint_transforms[idx][:3, 3] += delta
                self.update_children_cascade(idx, delta)
            self._latest_state_version += 1
            self._pending_compute = True

        if self._pending_compute and not self._compute_inflight:
            self._dispatch_compute()

    def _dispatch_compute(self) -> None:
        if self.mesh is None or self.skeleton is None:
            return
        self._pending_compute = False
        self._compute_inflight = True
        version = self._latest_state_version
        pose = [self.joint_transforms[j] for j in range(self.skeleton.n)]
        if self.skinning_mode == "simple":
            weights = self.simple_weights_topk or self.simple_weights
        else:
            weights = self.weights_topk or self.weights

        job = {
            "version": version,
            "compile_mode": self.compile_mode,
            "vertices": self.mesh.vertices,
            "skeleton": self.skeleton,
            "weights": weights,
            "pose": pose,
        }
        self.compute_requested.emit(job)

    def _on_deform_ready(self, vertices: np.ndarray, version: int) -> None:
        self._compute_inflight = False
        if version != self._latest_state_version:
            self._pending_compute = True
            return
        if self.viewport:
            self.viewport.update_deformed_mesh_only(vertices)

    def _on_deform_failed(self, msg: str, version: int) -> None:
        self._compute_inflight = False
        self._notify(f"Deform failed: {msg}")

    def _on_weights_ready(
        self,
        weights: np.ndarray,
        weights_topk,
        simple_weights: np.ndarray,
        simple_weights_topk,
    ) -> None:
        self.weights = weights
        self.weights_topk = weights_topk
        self.simple_weights = simple_weights
        self.simple_weights_topk = simple_weights_topk
        self.weights_dirty = False
        self._weights_inflight = False
        self._latest_state_version += 1
        self._pending_compute = True
        self._notify("Weights recomputed.")

    def _on_weights_failed(self, msg: str) -> None:
        self._weights_inflight = False
        self._notify(f"Weights computation failed: {msg}")
    # --------------------------------------------------------------- UI hooks

    def get_transforms(self):
        return self.joint_transforms

    def set_transforms(self, transforms):
        self.joint_transforms = transforms
        self._latest_state_version += 1
        self._pending_compute = True

    def update_viewport_deformed(self):
        self.request_deform_update()

    def update_viewport_full(self):
        if self.viewport:
            self.viewport.render_scene_full()

    def request_deform_update(self) -> None:
        self._pending_compute = True

    def _notify(self, msg: str):
        self.statusBar().showMessage(msg)
        print(msg)


def run():
    app = QApplication(sys.argv)
    apply_style(app)
    window = SpotRigWindow()
    window.show()
    sys.exit(app.exec())
