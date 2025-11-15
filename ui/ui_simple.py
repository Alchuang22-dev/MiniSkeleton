# -*- coding: utf-8 -*-
"""
Spot 模型骨架绑定 UI（基于 rigging/ 下的新算法）

功能：
- 使用 data/single/spot/spot_control_mesh.obj 作为测试模型
- 使用 Skeleton + quadruped_auto_place_from_bbox 自动生成四足骨架
- 使用 Pinocchio 风格的 heat weights 作为“完整蒙皮”
- 使用最近关节 1-hot 作为“简化蒙皮”
- 支持鼠标点击关节、拖拽关节（及其子关节）进行交互式变形预览
"""

from rigging.mesh_io import Mesh
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QComboBox,
    QGroupBox, QSplitter
)
from PyQt5.QtCore import Qt, QEvent, QTimer
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from rigging.gltf_loader import load_skeleton_from_glb

# === 新的 rigging 模块 ===
from rigging.mesh_io import load_mesh
from rigging.skeleton import Skeleton, quadruped_auto_place_from_bbox
from rigging.weights_heat import compute_heat_weights, HeatWeightsConfig
from rigging.lbs import linear_blend_skinning


class SpotRigUI(QMainWindow):
    """Spot 模型骨架绑定 UI"""

    def __init__(self):
        super().__init__()

        # 数据存储
        self.mesh = None                 # rigging.mesh_io.Mesh
        self.skeleton: Skeleton = None   # 自动生成的四足骨架
        self.bones = []                  # [(parent, child), ...]
        self.weights = None              # 完整权重（heat weights）
        self.simple_weights = None       # 简化权重（最近关节 1-hot）
        self.joint_transforms = None     # (J,4,4) 局部相对 bind 的增量
        self.initial_joint_transforms = None

        # 选中的关节
        self.selected_joint = None
        self.joint_sphere_actors = {}

        # 坐标轴箭头
        self.axis_arrows = {}
        self.dragging_axis = None

        # 拖拽状态
        self.is_dragging = False
        self.last_mouse_pos = None

        # 缓存 Actor
        self.mesh_actor = None
        self.bone_actors = []
        self.joint_actors = []
        self.gizmo_actors = []
        self.label_actor = None

        # 延迟更新
        self.pending_update = False
        self.update_timer = QTimer()
        self.update_timer.setInterval(16)  # ~60 FPS
        self.update_timer.timeout.connect(self._deferred_update)

        # 蒙皮模式：'full'（heat weights） 或 'simple'（最近关节）
        self.skinning_mode = 'full'

        self.init_ui()
        self.load_model()

        # 可选：补一下法线，方便可视化
        if self.mesh is not None:
            self.mesh.ensure_vertex_normals(recompute=True)

    # ---------------- UI 初始化 ----------------

    def init_ui(self):
        """初始化 UI"""
        self.setWindowTitle("Spot 骨架绑定工具（Heat Weights + LBS）")
        self.setGeometry(100, 100, 1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧工具栏
        toolbar_widget = self.create_toolbar()

        # 右侧 3D 视图
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(toolbar_widget)
        splitter.addWidget(self.plotter.interactor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 1150])

        main_layout.addWidget(splitter)

        # 事件过滤器
        self.plotter.interactor.installEventFilter(self)

        # VTK picker
        self.picker = vtk.vtkPropPicker()

        self.statusBar().showMessage("💡 点击红色球体选择关节，拖拽箭头沿轴移动")

    def create_toolbar(self):
        """创建左侧工具栏"""
        toolbar = QWidget()
        toolbar.setFixedWidth(250)
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #000000;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#resetButton {
                background-color: #ff9800;
            }
            QPushButton#resetButton:hover {
                background-color: #e68900;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
                color: #333;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #666;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #333;
                selection-background-color: #4CAF50;
                selection-color: white;
                border: 1px solid #cccccc;
            }
        """)

        layout = QVBoxLayout(toolbar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 标题
        title = QLabel("Spot 骨架绑定工具")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # 控制组
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()

        self.reset_button = QPushButton("🔄 重置到初始状态")
        self.reset_button.setObjectName("resetButton")
        self.reset_button.clicked.connect(self.reset_to_initial)
        control_layout.addWidget(self.reset_button)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 蒙皮设置组
        skinning_group = QGroupBox("蒙皮设置")
        skinning_layout = QVBoxLayout()

        mode_label = QLabel("蒙皮模式:")
        mode_label.setStyleSheet("font-weight: normal; color: #555;")
        skinning_layout.addWidget(mode_label)

        self.skinning_combo = QComboBox()
        self.skinning_combo.addItem("完整蒙皮（Heat 权重 / Pinocchio）", "full")
        self.skinning_combo.addItem("简化蒙皮（最近关节 1-hot）", "simple")
        self.skinning_combo.currentIndexChanged.connect(self.on_skinning_mode_changed)
        skinning_layout.addWidget(self.skinning_combo)

        mode_info = QLabel(
            "• 完整蒙皮：基于 heat kernel 的平滑权重\n"
            "• 简化蒙皮：每个顶点只跟随最近关节"
        )
        mode_info.setStyleSheet(
            "font-size: 11px; color: #555; "
            "background-color: #fff; padding: 8px; "
            "border-radius: 3px; border: 1px solid #ddd;"
        )
        mode_info.setWordWrap(True)
        skinning_layout.addWidget(mode_info)

        skinning_group.setLayout(skinning_layout)
        layout.addWidget(skinning_group)

        layout.addStretch()

        info_label = QLabel("Spot Demo · Heat Weights + LBS")
        info_label.setStyleSheet("font-size: 10px; color: #999;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        return toolbar

    # ---------------- Model / Skeleton / Weights ----------------

    def load_model(self):
        try:
            glb_path = "data/single/spot/spot.glb"

            print("\n==================== [STEP 1] LOAD MESH + SKELETON FROM GLB ====================")
            print(f"📦 从 GLB 读取高模 + 骨架: {glb_path}")

            # 新：一次性从 glb 获得 vertices / faces / skeleton
            from rigging.gltf_loader import load_mesh_and_skeleton_from_glb
            verts, faces, names, parents, joint_positions = load_mesh_and_skeleton_from_glb(glb_path)

            print(f"  ▶ glb vertices: {verts.shape}")
            print(f"  ▶ glb faces   : {faces.shape}")

            # 用 glb 的高模构造 Mesh（不再用 spot_control_mesh.obj）
            self.mesh = Mesh(
                vertices=verts.astype(np.float32),
                faces=faces.astype(np.int32),
            )
            self.mesh.ensure_vertex_normals(recompute=True)

            V = self.mesh.vertices
            mesh_aabb_min = V.min(axis=0)
            mesh_aabb_max = V.max(axis=0)
            mesh_center = (mesh_aabb_min + mesh_aabb_max) * 0.5
            mesh_scale = np.linalg.norm(mesh_aabb_max - mesh_aabb_min)

            print(f"  ▶ mesh AABB min: {mesh_aabb_min}")
            print(f"  ▶ mesh AABB max: {mesh_aabb_max}")
            print(f"  ▶ mesh center  : {mesh_center}")
            print(f"  ▶ mesh scale   : {mesh_scale}")

            # ==================== [STEP 2] BUILD SKELETON ====================
            print("\n==================== [STEP 2] BUILD SKELETON ====================")
            self.skeleton = Skeleton.from_bind_positions(names, parents, joint_positions)
            print(f"  ▶ Skeleton 构建完成: {self.skeleton.n} joints")

            # 记录骨骼连线（parent-child）
            self.bones = [
                (j.parent, i)
                for i, j in enumerate(self.skeleton.joints)
                if j.parent is not None and j.parent >= 0
            ]
            print(f"  ▶ bones (edges): {len(self.bones)} 条")

            # ==================== [STEP 3] FK 检查 ====================
            bind_locals = [j.bind_local for j in self.skeleton.joints]
            G_bind = self.skeleton.forward_kinematics_local(bind_locals)

            print(f"  ▶ G_bind shape: {G_bind.shape}")
            print(f"  ▶ G_bind joints (first 5 positions):\n{G_bind[:5, :3, 3]}")

            fk_center = G_bind[:, :3, 3].mean(axis=0)
            print(f"  ▶ FK joint center     : {fk_center}")
            print(f"  ▶ FK - Mesh center    : {fk_center - mesh_center}")

            # ==================== [STEP 4] HEAT WEIGHTS ====================
            print("\n==================== [STEP 4] HEAT WEIGHTS ====================")
            from rigging.weights_heat import HeatWeightsConfig, compute_heat_weights

            cfg = HeatWeightsConfig(
                tau=0.5,
                topk=4,
                smooth_passes=1,
            )
            print("🔥 计算 Heat 权重（Pinocchio-style）...")
            self.weights = compute_heat_weights(self.mesh, self.skeleton, cfg)
            print("  ▶ Heat weights shape:", self.weights.shape)

            # ==================== [STEP 5] SIMPLE WEIGHTS ====================
            print("\n==================== [STEP 5] SIMPLE WEIGHTS ====================")
            joint_positions_fk = G_bind[:, :3, 3]
            self.simple_weights = self.compute_simple_weights(self.mesh.vertices, joint_positions_fk)
            print("  ▶ Simple weights computed")

            # ==================== [STEP 6] INIT TRANSFORMS ====================
            print("\n==================== [STEP 6] INIT TRANSFORMS ====================")
            J = self.skeleton.n
            self.joint_transforms = np.eye(4)[None, :, :].repeat(J, axis=0)
            self.initial_joint_transforms = self.joint_transforms.copy()
            print("  ▶ transforms initialized")

            # ==================== [STEP 7] RENDER ====================
            print("\n==================== [STEP 7] RENDER ====================")
            self.render_scene_full()

            self.statusBar().showMessage(
                f"✅ Spot(glb) 加载成功：{self.skeleton.n} 个关节, {self.mesh.n_vertices} 顶点"
            )

        except Exception as e:
            print("加载失败：", e)
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"❌ 加载失败：{e}")


    @staticmethod
    def compute_simple_weights(vertices, joint_positions):
        """简化权重：每个顶点只跟随最近的关节（1-hot）"""
        vertices = np.asarray(vertices, dtype=np.float32)
        joint_positions = np.asarray(joint_positions, dtype=np.float32)

        n_verts = vertices.shape[0]
        n_joints = joint_positions.shape[0]

        distances = np.linalg.norm(
            vertices[:, None, :] - joint_positions[None, :, :],
            axis=2
        )  # (N,J)

        nearest_joint = np.argmin(distances, axis=1)

        weights = np.zeros((n_verts, n_joints), dtype=np.float32)
        weights[np.arange(n_verts), nearest_joint] = 1.0

        print("✅ 简化权重计算完成")
        return weights

    # ---------------- 状态 / 事件 ----------------

    def reset_to_initial(self):
        """重置到初始 bind 姿态"""
        if self.initial_joint_transforms is None:
            self.statusBar().showMessage("⚠️ 没有可重置的初始状态")
            return

        self.joint_transforms = self.initial_joint_transforms.copy()
        self.selected_joint = None
        self.update_deformed_mesh_only()

        self.statusBar().showMessage("✅ 已重置到初始状态")
        print("🔄 重置到初始状态")

    def on_skinning_mode_changed(self, index):
        """蒙皮模式切换"""
        self.skinning_mode = self.skinning_combo.itemData(index)
        self.update_deformed_mesh_only()

        mode_name = self.skinning_combo.currentText()
        self.statusBar().showMessage(f"✅ 切换到：{mode_name}")
        print(f"🎨 蒙皮模式切换为：{self.skinning_mode}")

    # ---------------- 事件过滤 / 鼠标交互 ----------------

    def eventFilter(self, obj, event):
        if obj == self.plotter.interactor:
            if event.type() == QEvent.MouseButtonPress:
                self.handle_mouse_press(event)
                return False
            elif event.type() == QEvent.MouseMove:
                self.handle_mouse_move(event)
                return self.is_dragging
            elif event.type() == QEvent.MouseButtonRelease:
                self.handle_mouse_release(event)
                return False

        return super().eventFilter(obj, event)

    def handle_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            mouse_x, mouse_y = event.x(), event.y()

            window_size = self.plotter.window_size
            dpr = self.plotter.interactor.devicePixelRatio()

            mouse_x_scaled = mouse_x * dpr
            mouse_y_scaled = mouse_y * dpr
            window_height = window_size[1]

            self.picker.Pick(mouse_x_scaled, window_height - mouse_y_scaled, 0, self.plotter.renderer)
            picked_actor = self.picker.GetActor()

            if picked_actor is not None:
                # 1) 拖拽 gizmo 轴
                if picked_actor in self.axis_arrows:
                    axis_name, axis_vector = self.axis_arrows[picked_actor]
                    self.is_dragging = True
                    self.dragging_axis = (axis_name, axis_vector)
                    self.last_mouse_pos = (mouse_x, mouse_y)
                    self.plotter.disable()
                    print(f"🎯 开始拖拽 {axis_name.upper()} 轴")
                    return

                # 2) 拾取关节球
                for sphere_actor, joint_idx in self.joint_sphere_actors.items():
                    if sphere_actor == picked_actor:
                        if self.selected_joint == joint_idx:
                            # 再次点击 -> 拖拽模式
                            self.is_dragging = True
                            self.last_mouse_pos = (mouse_x, mouse_y)
                            self.plotter.disable()
                            print(f"🖱️ 开始拖拽关节 [{joint_idx}]")
                        else:
                            self.selected_joint = joint_idx
                            self.update_gizmo_only()
                            joint_name = self.skeleton.joints[joint_idx].name
                            self.statusBar().showMessage(
                                f"✅ 选中关节 [{joint_idx}] {joint_name}"
                            )
                            print(f"✅ 选中关节 [{joint_idx}] {joint_name}")
                        return

                # 3) 点击其他地方 -> 取消选中
                if self.selected_joint is not None:
                    self.selected_joint = None
                    self.update_gizmo_only()
                    self.statusBar().showMessage("💡 点击红色球体选择关节")
            else:
                if self.selected_joint is not None:
                    self.selected_joint = None
                    self.update_gizmo_only()
                    self.statusBar().showMessage("💡 点击红色球体选择关节")

    def handle_mouse_move(self, event):
        if self.is_dragging and event.buttons() & Qt.LeftButton and self.selected_joint is not None:
            x, y = event.x(), event.y()

            if self.last_mouse_pos is None:
                self.last_mouse_pos = (x, y)
                return

            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]

            if abs(dx) < 1 and abs(dy) < 1:
                return

            camera = self.plotter.camera
            camera_pos = np.array(camera.GetPosition())

            # 当前关节的世界位置（GLOBAL）
            G_current = self.compute_current_global_mats()
            joint_pos = G_current[self.selected_joint, :3, 3]

            distance = np.linalg.norm(camera_pos - joint_pos)
            scale = distance * 0.001

            if self.dragging_axis is not None:
                axis_name, axis_vector = self.dragging_axis
                view_up = np.array(camera.GetViewUp())
                view_dir = camera_pos - joint_pos
                view_dir = view_dir / np.linalg.norm(view_dir)

                right = np.cross(view_up, view_dir)
                right = right / np.linalg.norm(right)
                up = np.cross(view_dir, right)
                up = up / np.linalg.norm(up)

                screen_delta = right * dx * scale + up * dy * scale
                delta = np.dot(screen_delta, axis_vector) * axis_vector
            else:
                view_up = np.array(camera.GetViewUp())
                view_dir = camera_pos - joint_pos
                view_dir = view_dir / np.linalg.norm(view_dir)

                right = np.cross(view_up, view_dir)
                right = right / np.linalg.norm(right)
                up = np.cross(view_dir, right)
                up = up / np.linalg.norm(up)

                delta = right * dx * scale + up * dy * scale

            # 修改局部增量平移，并级联到子关节
            self.joint_transforms[self.selected_joint][:3, 3] += delta
            self.update_children_cascade(self.selected_joint, delta)

            self.last_mouse_pos = (x, y)

            self.pending_update = True
            if not self.update_timer.isActive():
                self.update_timer.start()

    def handle_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.dragging_axis = None
            self.last_mouse_pos = None
            self.plotter.enable()

            self.update_timer.stop()
            self.update_deformed_mesh_only()

            if self.selected_joint is not None:
                joint_name = self.skeleton.joints[self.selected_joint].name
                self.statusBar().showMessage(
                    f"✅ 关节 [{self.selected_joint}] {joint_name} 移动完成"
                )
                print("✅ 拖拽完成")

    def _deferred_update(self):
        if self.pending_update:
            self.pending_update = False
            self.update_deformed_mesh_only()

    # ---------------- 骨架层级辅助 ----------------

    def get_joint_children(self, joint_idx):
        children = []
        for i, joint in enumerate(self.skeleton.joints):
            if joint.parent == joint_idx:
                children.append(i)
        return children

    def update_children_cascade(self, parent_idx, delta):
        """递归平移所有子关节的局部增量"""
        children = self.get_joint_children(parent_idx)
        for child_idx in children:
            self.joint_transforms[child_idx][:3, 3] += delta
            self.update_children_cascade(child_idx, delta)

    def compute_current_global_mats(self):
        """基于当前局部增量 joint_transforms 计算全局矩阵 G_current"""
        pose = [self.joint_transforms[j] for j in range(self.skeleton.n)]
        G_current = self.skeleton.forward_kinematics_pose(pose)
        return G_current

    # ---------------- LBS / 变形计算 ----------------

    def compute_deformed_vertices(self):
        """使用当前关节姿态和选定权重计算变形后的顶点"""
        pose = [self.joint_transforms[j] for j in range(self.skeleton.n)]

        # 生成 skinning matrices: M_skin[j] = G_current[j] @ inv_bind[j]
        M_skin = self.skeleton.skinning_matrices(pose)  # (J,4,4)

        # 根据模式选择权重
        if self.skinning_mode == 'simple':
            weights = self.simple_weights
        else:
            weights = self.weights

        deformed_vertices = linear_blend_skinning(
            self.mesh.vertices,
            weights,
            M_skin,
            topk=None,       # 权重已经经过 top-k 和归一化，无需再次处理
            normalize=False
        )

        return deformed_vertices

    # ---------------- 渲染相关 ----------------

    def render_scene_full(self):
        """完整渲染场景（网格 + 骨骼 + 关节 + Gizmo）"""
        self.plotter.clear()
        self.joint_sphere_actors = {}
        self.axis_arrows = {}
        self.bone_actors = []
        self.joint_actors = []

        G_current = self.compute_current_global_mats()
        current_joint_positions = G_current[:, :3, 3]

        deformed_vertices = self.compute_deformed_vertices()

        mesh_size = np.linalg.norm(
            deformed_vertices.max(axis=0) - deformed_vertices.min(axis=0)
        )
        sphere_radius = mesh_size * 0.015

        # 1. 网格
        faces = self.mesh.faces.astype(np.int64)
        faces_with_count = np.hstack([np.full((len(faces), 1), 3), faces])
        mesh_pv = pv.PolyData(deformed_vertices, faces_with_count)
        self.mesh_actor = self.plotter.add_mesh(
            mesh_pv,
            color='lightblue',
            opacity=0.6,
            show_edges=True,
            edge_color='navy',
            line_width=0.3,
            smooth_shading=True,
            pickable=False
        )

        # 2. 骨骼（线段）
        for jp, jc in self.bones:
            p1 = current_joint_positions[jp]
            p2 = current_joint_positions[jc]
            line = pv.Line(p1, p2)
            actor = self.plotter.add_mesh(
                line,
                color='darkred',
                line_width=8,
                opacity=0.8,
                pickable=False
            )
            self.bone_actors.append((actor, jp, jc))

        # 3. 关节球
        for i, pos in enumerate(current_joint_positions):
            sphere = pv.Sphere(
                radius=sphere_radius,
                center=pos.tolist(),
                theta_resolution=16,
                phi_resolution=16
            )
            color = 'yellow' if i == self.selected_joint else 'red'
            actor = self.plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.9,
                pickable=True,
                lighting=True
            )
            self.joint_sphere_actors[actor] = i
            self.joint_actors.append((actor, i, sphere_radius))

        # 4. Gizmo
        self.update_gizmo_only()

        # 5. 相机
        if not hasattr(self, '_camera_set'):
            self.plotter.reset_camera()
            self.plotter.camera.elevation = 15
            self.plotter.camera.azimuth = -60
            self.plotter.camera.zoom(1.4)
            self._camera_set = True

        self.plotter.update()

    def update_deformed_mesh_only(self):
        """只更新网格顶点和骨骼位置（不重建 Actor）"""
        if self.mesh_actor is None:
            return

        deformed_vertices = self.compute_deformed_vertices()

        vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
        vtk_array = numpy_to_vtk(deformed_vertices, deep=True)
        vtk_points.SetData(vtk_array)
        vtk_points.Modified()

        G_current = self.compute_current_global_mats()
        current_joint_positions = G_current[:, :3, 3]

        # 更新骨骼线
        for actor, jp, jc in self.bone_actors:
            p1 = current_joint_positions[jp]
            p2 = current_joint_positions[jc]
            line = pv.Line(p1, p2)
            actor.GetMapper().SetInputData(line)

        # 更新关节球
        for actor, joint_idx, radius in self.joint_actors:
            pos = current_joint_positions[joint_idx]
            sphere = pv.Sphere(
                radius=radius,
                center=pos.tolist(),
                theta_resolution=16,
                phi_resolution=16
            )
            actor.GetMapper().SetInputData(sphere)

        self.update_gizmo_only()
        self.plotter.update()

    def update_gizmo_only(self):
        """只更新 Gizmo（箭头 + 标注）"""
        for actor in self.gizmo_actors:
            self.plotter.remove_actor(actor)
        self.gizmo_actors = []
        self.axis_arrows = {}

        if self.label_actor is not None:
            self.plotter.remove_actor(self.label_actor)
            self.label_actor = None

        if self.selected_joint is None:
            self.plotter.update()
            return

        G_current = self.compute_current_global_mats()
        current_joint_positions = G_current[:, :3, 3]
        pos = current_joint_positions[self.selected_joint]

        mesh_size = np.linalg.norm(
            self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)
        )
        arrow_length = mesh_size * 0.1

        axes = [
            ('x', np.array([1.0, 0.0, 0.0]), 'red'),
            ('y', np.array([0.0, 1.0, 0.0]), 'green'),
            ('z', np.array([0.0, 0.0, 1.0]), 'blue')
        ]

        for axis_name, direction, color in axes:
            arrow = pv.Arrow(
                start=pos.tolist(),
                direction=direction.tolist(),
                tip_length=0.25,
                tip_radius=0.1,
                shaft_radius=0.03,
                scale=float(arrow_length)
            )
            actor = self.plotter.add_mesh(
                arrow,
                color=color,
                opacity=0.8,
                pickable=True,
                lighting=True
            )
            self.axis_arrows[actor] = (axis_name, direction)
            self.gizmo_actors.append(actor)

        joint_name = self.skeleton.joints[self.selected_joint].name
        sphere_radius = mesh_size * 0.015
        label_pos = pos + np.array([0, sphere_radius * 3, 0])

        self.label_actor = self.plotter.add_point_labels(
            [label_pos],
            [f"[{self.selected_joint}] {joint_name}"],
            font_size=14,
            bold=True,
            text_color='black',
            point_color='yellow',
            point_size=20,
            shape_opacity=0.8
        )

        self.plotter.update()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SpotRigUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
