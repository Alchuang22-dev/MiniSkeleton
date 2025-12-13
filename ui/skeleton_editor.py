"""Main rig editor window and controller."""

from __future__ import annotations

import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QSplitter, QWidget

from rigging.mesh_io import Mesh
from rigging.gltf_loader import load_mesh_and_skeleton_from_glb
from rigging.skeleton import Skeleton
from rigging.lbs import linear_blend_skinning
from rigging.weights_heat import HeatWeightsConfig, compute_heat_weights

from ui.action_timeline import ActionTimeline
from ui.export_panel import RigControlPanel
from ui.viewport import RigViewport
from ui.weight_tools import compute_simple_weights


class SpotRigWindow(QMainWindow):
    """Spot model rig binding demo with a small keyframe timeline."""

    def __init__(self):
        super().__init__()

        # Core data
        self.mesh: Mesh | None = None
        self.skeleton: Skeleton | None = None
        self.bones: list[tuple[int, int]] = []
        self.weights = None
        self.simple_weights = None
        self.joint_transforms = None  # (J,4,4)
        self.initial_joint_transforms = None

        # Interaction state
        self.selected_joint: int | None = None
        self.skinning_mode = "full"

        # UI components
        self.timeline: ActionTimeline | None = None
        self.toolbar: RigControlPanel | None = None
        self.viewport: RigViewport | None = None

        self._init_ui()
        self.load_model()

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
            on_update_mesh=self.update_viewport_deformed,
            on_status=self._notify,
        )
        self.toolbar = RigControlPanel(
            timeline_widget=self.timeline,
            on_reset=self.reset_to_initial,
            on_skinning_change=self.on_skinning_mode_changed,
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

    # ---------------------------------------------------------------- Loading

    def load_model(self):
        """Load spot.glb mesh + skeleton and initialize weights/pose."""
        try:
            glb_path = "data/single/spot/spot.glb"
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

            print("\n==================== [STEP 5] SIMPLE WEIGHTS ====================")
            joint_positions_fk = G_bind[:, :3, 3]
            self.simple_weights = compute_simple_weights(self.mesh.vertices, joint_positions_fk)
            print("  - Simple weights computed")

            print("\n==================== [STEP 6] INIT TRANSFORMS ====================")
            J = self.skeleton.n
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(J, axis=0)
            self.initial_joint_transforms = self.joint_transforms.copy()
            if self.timeline:
                self.timeline.reset()
            print("  - transforms initialized")

            print("\n==================== [STEP 7] RENDER ====================")
            self.update_viewport_full()
            self._notify(f"Spot(glb) loaded: {self.skeleton.n} joints, {self.mesh.n_vertices} vertices")
        except Exception as e:  # noqa: BLE001
            print("Load failed:", e)
            import traceback

            traceback.print_exc()
            self._notify(f"Load failed: {e}")

    # -------------------------------------------------------------- Callbacks

    def reset_to_initial(self):
        if self.initial_joint_transforms is None:
            self._notify("No initial pose to reset.")
            return
        self.set_transforms(np.array(self.initial_joint_transforms, copy=True))
        self.selected_joint = None
        self.update_viewport_deformed()
        self._notify("Reset to bind pose.")

    def on_skinning_mode_changed(self, mode: str):
        self.skinning_mode = mode
        self.update_viewport_deformed()
        self._notify(f"Switched skinning mode to: {mode}")

    # --------------------------------------------------------------- Geometry

    def compute_deformed_vertices(self):
        pose = [self.joint_transforms[j] for j in range(self.skeleton.n)]
        M_skin = self.skeleton.skinning_matrices(pose)
        weights = self.simple_weights if self.skinning_mode == "simple" else self.weights
        return linear_blend_skinning(
            self.mesh.vertices,
            weights,
            M_skin,
            topk=None,
            normalize=False,
        )

    def compute_current_global_mats(self):
        pose = [self.joint_transforms[j] for j in range(self.skeleton.n)]
        return self.skeleton.forward_kinematics_pose(pose)

    def get_joint_children(self, joint_idx):
        children = []
        for i, joint in enumerate(self.skeleton.joints):
            if joint.parent == joint_idx:
                children.append(i)
        return children

    def update_children_cascade(self, parent_idx, delta):
        children = self.get_joint_children(parent_idx)
        for child_idx in children:
            self.joint_transforms[child_idx][:3, 3] += delta
            self.update_children_cascade(child_idx, delta)

    # --------------------------------------------------------------- UI hooks

    def get_transforms(self):
        return self.joint_transforms

    def set_transforms(self, transforms):
        self.joint_transforms = transforms

    def update_viewport_deformed(self):
        if self.viewport:
            self.viewport.update_deformed_mesh_only()

    def update_viewport_full(self):
        if self.viewport:
            self.viewport.render_scene_full()

    def _notify(self, msg: str):
        self.statusBar().showMessage(msg)
        print(msg)


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SpotRigWindow()
    window.show()
    sys.exit(app.exec_())
