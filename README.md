# MiniSkeleton
动画大作业暂定库

目录结构：

```
animation-skeleton/
├─ data/                         # 放模型（已下载）
│  ├─ single/                    # 单模型测试
│  └─ multi/                     # 多模型交互测试
├─ out/                          # 渲染帧与视频输出
│  ├─ frames/
│  └─ videos/
├─ rigging/                      # 算法核心
│  ├─ __init__.py
│  ├─ mesh_io.py                 # OBJ/PLY 读写、拓扑检查、法线/UV修复
│  ├─ skeleton.py                # 关节/骨骼数据结构 + FK（含层级/约束）
│  ├─ weights_nearest.py         # 最近骨骼/双骨插值权重（基础必做）
│  ├─ gltf_loader.py             # GLB读写
│  ├─ weights_heat.py            # 热扩散/拉普拉斯权重（进阶）
│  ├─ lbs.py                     # 线性混合蒙皮 (Linear Blend Skinning)
│  ├─ deformation_transfer.py    # （可选）DT 动画复用
│  └─ constraints.py             # （可选）IK/关节限制/姿态约束
├─ scene/
│  ├─ __init__.py
│  ├─ asset.py                   # Model+Skeleton+Weights 的资源封装
│  ├─ timeline.py                # 动画轨/关键帧/曲线（Bezier/Ease）
│  ├─ actions.py                 # 姿态库、程序化动作、混合（blend）
│  └─ choreography.py            # 多模型交互编排（时序/碰撞占位）
├─ ui/                           # 可视化 UI（基于 PySide6）
│  ├─ __init__.py
│  ├─ app.py                     # QApplication / 主窗口
│  ├─ viewport.py                # 3D 视口（OpenGL / moderngl 嵌入）
│  ├─ skeleton_editor.py         # 1.1/1.2：骨架关节创建/连线/编辑/对齐
│  ├─ weight_tools.py            # 1.3：自动权重计算/可视化/刷权
│  ├─ action_timeline.py         # 1.4：动作轨编辑、回放、导出
│  └─ export_panel.py            # 视频导出（mp4/gif）
├─ render/
│  ├─ __init__.py
│  ├─ offscreen_mgl.py           # 用 moderngl 离屏渲染为 PNG
│  ├─ onscreen_qt.py             # UI 视口渲染桥接（与 ui/viewport.py 配合）
│  └─ make_video.sh              # 调 ffmpeg 合成 mp4
├─ tools/
│  ├─ preview_obj.py             # 预览网格，检查拓扑
│  ├─ bake_frames.py             # 批量离屏渲染帧
│  └─ export_video.py            # 把帧合成 mp4/gif
├─ tests/                        # 关键模块的最小单测
│  ├─ test_mesh_io.py
│  ├─ test_skeleton_fk.py
│  ├─ test_weights.py
│  └─ test_lbs.py
├─ examples/
│  ├─ single_model_demo.py       # 一键：加载→放置骨架→算权重→播放→渲染
│  └─ multi_model_demo.py        # 多模型交互示例
├─ main_demo.py                  # 同 single_model_demo，留作入口
├─ requirements.txt
└─ README.md
```