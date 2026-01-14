"""Main rig editor window and controller."""

from __future__ import annotations

import csv
import glob
import os
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
from rigging.skeleton import Skeleton, euler_xyz_to_rot
from rigging.lbs import TopKWeights, linear_blend_skinning, make_topk_weights
from rigging.weights_heat import HeatWeightsConfig, compute_heat_weights
from rigging.rig_io import save_rig_npz
from rigging.skeleton_optimize import optimize_quadruped_bind_positions
from rigging.gltf_export import export_skeleton_glb as gltf_export_skeleton
from render.offscreen_vtk import OffscreenVtkRenderer
from tools.export_video import frames_to_video

from ui.action_timeline import ActionTimeline
from ui.auto_motion import (
    DEMO_DURATION,
    DEMO_FPS,
    DEMO_HEAD_YAW,
    DEMO_NECK_YAW,
    WALK_BASE_DURATION,
    WALK_LEG_SWING,
    WALK_REPEAT,
    build_head_shake_keyframes,
    build_walk_keyframes,
)
from ui.compute_worker import DeformWorker, WeightsWorker
from ui.export_panel import RigControlPanel
from ui.viewport import RigViewport
from ui.weight_tools import compute_simple_weights
from ui.style import apply_style

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sp = None

RENDER_YAW_DEG = -90.0


def _rotation_y(angle_rad: float) -> np.ndarray:
    R = euler_xyz_to_rot(0.0, float(angle_rad), 0.0)
    return R.astype(np.float32)


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
        self._bind_locals: list[np.ndarray] | None = None
        self._orig_joint_names: list[str] = []
        self._orig_joint_parents: list[int] = []
        self._orig_bind_positions: np.ndarray | None = None
        self._orig_bind_locals: list[np.ndarray] | None = None
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
        self.model_offset = np.zeros(3, dtype=np.float32)

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
            on_generate_demo=self.generate_demo_keyframes,
            on_generate_walk=self.generate_walk_keyframes,
        )
        self.toolbar = RigControlPanel(
            timeline_widget=self.timeline,
            on_reset=self.reset_to_initial,
            on_skinning_change=self.on_skinning_mode_changed,
            on_load_model=self.load_model_from_path,
            on_toggle_compile=self.set_compile_mode,
            on_add_joint=self.add_joint,
            on_set_parent=self.reparent_selected_joint,
            on_clear_parent=self.clear_parent_selected_joint,
            on_recompute_weights=self.recompute_weights,
            on_reset_bind=self.reset_bind_pose,
            on_optimize_quadruped=self.optimize_quadruped_skeleton,
            on_export_skeleton=self.export_skeleton_glb,
            on_render_frames=self.render_frames_vtk,
            on_make_video=self.make_video_from_frames,
            on_save_rig=self.save_rig_bundle,
            on_export_joint_positions=self.export_joint_positions,
            on_export_bone_edges=self.export_skeleton_edges,
            on_export_weights=self.export_bind_weights,
            on_set_model_offset=self.set_model_offset,
            on_reset_model_offset=self.reset_model_offset,
        )
        self.viewport = RigViewport(self, self._notify)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.addWidget(self.toolbar)
        splitter.addWidget(self.viewport)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 880])
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

            (
                verts,
                faces,
                names,
                parents,
                joint_positions,
                bind_mats,
                skin_weights,
            ) = load_mesh_and_skeleton_from_glb(
                glb_path,
                return_bind_mats=True,
                return_weights=True,
            )
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
            if bind_mats is not None:
                self.skeleton = Skeleton.from_bind_matrices(names, parents, bind_mats)
            else:
                self.skeleton = Skeleton.from_bind_positions(names, parents, joint_positions)
            print(f"  - Skeleton joints: {self.skeleton.n}")
            self._bind_locals = [np.array(j.bind_local, copy=True) for j in self.skeleton.joints]
            self._orig_bind_locals = [np.array(j.bind_local, copy=True) for j in self.skeleton.joints]
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

            print("\n==================== [STEP 4] WEIGHTS ====================")
            if skin_weights is not None:
                self.weights = skin_weights
                topk = min(4, self.skeleton.n)
                self.weights_topk = make_topk_weights(self.weights, topk)
                self.weights_dirty = False
                print("  - Using GLB skin weights")
            else:
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
                base = os.path.splitext(os.path.basename(glb_path))[0]
                self.toolbar.set_video_paths(
                    os.path.join("out", "frames", base),
                    os.path.join("out", "videos", f"{base}.mp4"),
                )
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
        J = len(self.joint_names)
        if self._bind_locals is not None:
            if len(self._bind_locals) < J:
                missing = J - len(self._bind_locals)
                self._bind_locals.extend([np.eye(4, dtype=np.float32) for _ in range(missing)])
            elif len(self._bind_locals) > J:
                self._bind_locals = self._bind_locals[:J]

        if self._bind_locals is not None and len(self._bind_locals) == J:
            bind_locals = []
            for j in range(J):
                rot = self._bind_locals[j][:3, :3]
                if self.joint_parents[j] < 0:
                    t = self.bind_positions[j]
                else:
                    t = self.bind_positions[j] - self.bind_positions[self.joint_parents[j]]
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = rot
                mat[:3, 3] = t
                bind_locals.append(mat)

            bind_globals = np.zeros((J, 4, 4), dtype=np.float32)
            for j in range(J):
                p = self.joint_parents[j]
                if p < 0:
                    bind_globals[j] = bind_locals[j]
                else:
                    bind_globals[j] = bind_globals[p] @ bind_locals[j]

            self.skeleton = Skeleton.from_bind_matrices(
                self.joint_names,
                self.joint_parents,
                bind_globals,
            )
            self._bind_locals = bind_locals
        else:
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
        if self._bind_locals is not None:
            self._bind_locals.append(np.eye(4, dtype=np.float32))

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

    def clear_parent_selected_joint(self) -> None:
        if self.selected_joint is None:
            self._notify("Select a joint first.")
            return
        child_idx = self.selected_joint
        if self.joint_parents[child_idx] == -1:
            self._notify("Selected joint is already a root.")
            return
        self.joint_parents[child_idx] = -1
        self._rebuild_skeleton_from_bind()
        if self.toolbar:
            self.toolbar.set_joint_names(self.joint_names)
        self.update_viewport_full()
        self._notify(f"Cleared parent for joint [{child_idx}].")

    def reset_bind_pose(self) -> None:
        if self._orig_bind_positions is None:
            self._notify("No original bind pose available.")
            return
        self.joint_names = list(self._orig_joint_names)
        self.joint_parents = list(self._orig_joint_parents)
        self.bind_positions = np.array(self._orig_bind_positions, copy=True)
        if self._orig_bind_locals is not None:
            self._bind_locals = [np.array(m, copy=True) for m in self._orig_bind_locals]
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

    def render_frames_vtk(
        self,
        out_dir: str,
        fps: int,
        width: int,
        height: int,
        keyframes: list[np.ndarray] | None = None,
    ) -> None:
        if self.mesh is None or self.skeleton is None:
            self._notify("No mesh loaded; cannot render frames.")
            return
        if self.compile_mode:
            self._notify("Disable compile mode before rendering frames.")
            return
        if self.weights is None or self.simple_weights is None:
            self._notify("Weights missing; recompute weights first.")
            return

        os.makedirs(out_dir, exist_ok=True)

        if keyframes is None:
            keyframes = list(self.timeline.keyframes) if self.timeline else []
            if not keyframes:
                keyframes = [np.array(self.joint_transforms, copy=True)]

        camera_state = self._get_viewport_camera_state()
        use_viewport_camera = camera_state is not None
        renderer = OffscreenVtkRenderer(width=width, height=height)
        render_rot = None if use_viewport_camera else _rotation_y(np.radians(RENDER_YAW_DEG))
        offset = self.get_model_offset()
        renderer.set_mesh(
            self.mesh.vertices + offset,
            self.mesh.faces,
            rotation=render_rot,
            apply_ui_camera=not use_viewport_camera,
        )
        if use_viewport_camera:
            renderer.apply_camera_state(**camera_state)

        original_transforms = np.array(self.joint_transforms, copy=True)

        try:
            for idx, transforms in enumerate(keyframes):
                pose = np.array(transforms, copy=True)
                M_skin = self.skeleton.skinning_matrices(pose)
                if self.skinning_mode == "simple":
                    weights = self.simple_weights_topk or self.simple_weights
                else:
                    weights = self.weights_topk or self.weights
                deformed = linear_blend_skinning(
                    self.mesh.vertices,
                    weights,
                    M_skin,
                    topk=None,
                    normalize=False,
                )
                renderer.update_vertices(deformed + offset)
                out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
                renderer.render_to_file(out_path)
        finally:
            renderer.close()
            self.joint_transforms = original_transforms
            self._latest_state_version += 1
            self._pending_compute = True

        self._notify(f"Rendered {len(keyframes)} frames to {out_dir} at {fps} FPS.")

    def make_video_from_frames(
        self,
        frames_dir: str,
        video_path: str,
        fps: int,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if self.mesh is None or self.skeleton is None:
            self._notify("No mesh loaded; cannot export video.")
            return
        if self.compile_mode:
            self._notify("Disable compile mode before exporting video.")
            return

        frame_glob = os.path.join(frames_dir, "frame_*.png")
        if not glob.glob(frame_glob):
            self._notify("No frames found on disk; render frames first.")
            return
        try:
            frames_to_video(frames_dir, video_path, fps=fps)
            self._notify(f"Video saved: {video_path}")
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Video export failed: {exc}")

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

    def export_joint_positions(self) -> None:
        if self.skeleton is None:
            self._notify("No skeleton loaded; cannot export joints.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export joint positions",
            "out/debug/joints.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path = f"{path}.csv"
        names = self.joint_names or [j.name for j in self.skeleton.joints]
        parents = self.joint_parents or [j.parent for j in self.skeleton.joints]
        globals_mats = self.compute_current_global_mats()
        positions = globals_mats[:, :3, 3]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "name", "parent", "x", "y", "z"])
            for idx, pos in enumerate(positions):
                name = names[idx] if idx < len(names) else f"joint_{idx}"
                parent = parents[idx] if idx < len(parents) else -1
                writer.writerow(
                    [
                        idx,
                        name,
                        parent,
                        f"{pos[0]:.6f}",
                        f"{pos[1]:.6f}",
                        f"{pos[2]:.6f}",
                    ]
                )
        self._notify(f"Joint positions exported: {path}")

    def export_skeleton_edges(self) -> None:
        if self.skeleton is None:
            self._notify("No skeleton loaded; cannot export skeleton.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export skeleton edges",
            "out/debug/edges.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path = f"{path}.csv"
        names = self.joint_names or [j.name for j in self.skeleton.joints]
        parents = self.joint_parents or [j.parent for j in self.skeleton.joints]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parent_idx", "child_idx", "parent_name", "child_name"])
            for child_idx, parent_idx in enumerate(parents):
                if parent_idx < 0:
                    continue
                parent_name = names[parent_idx] if parent_idx < len(names) else f"joint_{parent_idx}"
                child_name = names[child_idx] if child_idx < len(names) else f"joint_{child_idx}"
                writer.writerow([parent_idx, child_idx, parent_name, child_name])
        self._notify(f"Skeleton edges exported: {path}")

    def export_bind_weights(self) -> None:
        if self.skeleton is None:
            self._notify("No skeleton loaded; cannot export weights.")
            return
        weights = self.weights_topk or self.weights
        simple = self.simple_weights_topk or self.simple_weights
        if weights is None and simple is None:
            self._notify("Weights missing; recompute weights first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export bind weights",
            "out/debug/weights.npz",
            "NPZ Files (*.npz)",
        )
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path = f"{path}.npz"
        names = self.joint_names or [j.name for j in self.skeleton.joints]
        parents = self.joint_parents or [j.parent for j in self.skeleton.joints]
        max_len = max((len(n) for n in names), default=1)
        payload: dict[str, np.ndarray | str] = {
            "joint_names": np.asarray(names, dtype=f"<U{max_len}"),
            "parents": np.asarray(parents, dtype=np.int32),
        }

        def _pack_weights(prefix: str, value) -> None:
            if value is None:
                return
            if isinstance(value, TopKWeights):
                payload[f"{prefix}_format"] = "topk"
                payload[f"{prefix}_indices"] = np.asarray(value.indices, dtype=np.int32)
                payload[f"{prefix}_values"] = np.asarray(value.weights, dtype=np.float32)
                return
            if sp is not None and sp.isspmatrix(value):
                payload[f"{prefix}_format"] = "csr"
                csr = value.tocsr()
                payload[f"{prefix}_data"] = np.asarray(csr.data, dtype=np.float32)
                payload[f"{prefix}_indices"] = np.asarray(csr.indices, dtype=np.int32)
                payload[f"{prefix}_indptr"] = np.asarray(csr.indptr, dtype=np.int32)
                payload[f"{prefix}_shape"] = np.asarray(csr.shape, dtype=np.int32)
                return
            if isinstance(value, tuple) and len(value) == 2:
                idxs, vals = value
                payload[f"{prefix}_format"] = "topk_tuple"
                payload[f"{prefix}_indices"] = np.asarray(idxs, dtype=np.int32)
                payload[f"{prefix}_values"] = np.asarray(vals, dtype=np.float32)
                return
            payload[f"{prefix}_format"] = "dense"
            payload[prefix] = np.asarray(value, dtype=np.float32)

        _pack_weights("weights", weights)
        _pack_weights("simple_weights", simple)
        np.savez_compressed(path, **payload)
        self._notify(f"Bind weights exported: {path}")

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

    def _get_export_settings(self):
        if self.toolbar is None:
            return None
        frames_edit = getattr(self.toolbar, "frames_path_edit", None)
        fps_spin = getattr(self.toolbar, "fps_spin", None)
        width_spin = getattr(self.toolbar, "width_spin", None)
        height_spin = getattr(self.toolbar, "height_spin", None)
        if frames_edit is None:
            return None
        out_dir = frames_edit.text().strip()
        fps = fps_spin.value() if fps_spin is not None else 30
        width = width_spin.value() if width_spin is not None else 1024
        height = height_spin.value() if height_spin is not None else 1024
        return out_dir, fps, width, height

    def _get_viewport_camera_state(self):
        if self.viewport is None:
            return None
        plotter = getattr(self.viewport, "plotter", None)
        cam = getattr(plotter, "camera", None) if plotter is not None else None
        if cam is None:
            return None
        return {
            "position": tuple(float(v) for v in cam.GetPosition()),
            "focal_point": tuple(float(v) for v in cam.GetFocalPoint()),
            "view_up": tuple(float(v) for v in cam.GetViewUp()),
            "view_angle": float(cam.GetViewAngle()),
            "clipping_range": tuple(float(v) for v in cam.GetClippingRange()),
        }

    def generate_demo_keyframes(self):
        if self.skeleton is None or self.joint_transforms is None:
            self._notify("No skeleton loaded; cannot generate demo keyframes.")
            return []
        base = np.array(self.joint_transforms, copy=True)
        names = self.joint_names or [j.name for j in self.skeleton.joints]
        keyframes = build_head_shake_keyframes(
            names,
            base,
            fps=DEMO_FPS,
            duration=DEMO_DURATION,
            head_yaw=DEMO_HEAD_YAW,
            neck_yaw=DEMO_NECK_YAW,
        )
        if not keyframes:
            self._notify("No head/neck joint found for demo.")
            return []

        self.set_transforms(np.array(keyframes[0], copy=True))
        self.update_viewport_deformed()
        export_settings = self._get_export_settings()
        if export_settings is None:
            self._notify("Set a frames output directory to save demo keyframes.")
            return keyframes
        out_dir, fps, width, height = export_settings
        if not out_dir:
            self._notify("Set a frames output directory to save demo keyframes.")
            return keyframes
        self._notify("Rendering demo keyframes to disk (UI pipeline)...")
        self.render_frames_vtk(out_dir, fps, width, height, keyframes=keyframes)
        return keyframes

    def generate_walk_keyframes(self):
        if self.skeleton is None or self.joint_transforms is None:
            self._notify("No skeleton loaded; cannot generate walking keyframes.")
            return []
        export_settings = self._get_export_settings()
        if export_settings is not None:
            out_dir, fps, width, height = export_settings
        else:
            out_dir, fps, width, height = "", DEMO_FPS, 1024, 1024

        if width < 512 or height < 512:
            width = max(width, 512)
            height = max(height, 512)
            self._notify(f"Walking export size clamped to {width}x{height}.")

        base = np.array(self.joint_transforms, copy=True)
        names = self.joint_names or [j.name for j in self.skeleton.joints]
        keyframes = build_walk_keyframes(
            names,
            base,
            fps=fps,
            duration=WALK_BASE_DURATION,
            repeat=WALK_REPEAT,
            leg_swing=WALK_LEG_SWING,
            close_loop=True,
        )
        if not keyframes:
            self._notify("No leg joints found for walking demo.")
            return []

        self.set_transforms(np.array(keyframes[0], copy=True))
        self.update_viewport_deformed()

        if not out_dir:
            self._notify("Set a frames output directory to save walking keyframes.")
            return keyframes

        self._notify("Rendering walking keyframes to disk (UI pipeline)...")
        self.render_frames_vtk(out_dir, fps, width, height, keyframes=keyframes)
        return keyframes
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

    def get_model_offset(self) -> np.ndarray:
        return np.array(self.model_offset, dtype=np.float32)

    def set_model_offset(self, x: float, y: float, z: float) -> None:
        self.model_offset = np.array([x, y, z], dtype=np.float32)
        if self.toolbar is not None:
            self.toolbar.set_model_offset((x, y, z))
        if self.viewport is not None:
            if getattr(self.viewport, "mesh_actor", None) is None:
                self.viewport.render_scene_full()
            else:
                self.viewport.update_deformed_mesh_only()

    def reset_model_offset(self) -> None:
        self.set_model_offset(0.0, 0.0, 0.0)

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
