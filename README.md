# MiniSkeleton

MiniSkeleton is a compact rigging and animation toolchain that loads GLB models,
builds/edits skeletons, computes skinning weights, previews deformation in a
PySide6 UI, and exports frame sequences and videos.

## Environment and Dependencies

Recommended environment:
- Windows 10/11 (tested on Windows)
- Python 3.10+ (64-bit)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

External tools:
- `ffmpeg` is required for `tools/export_video.py` and the UI "Make video" flow.
  Ensure `ffmpeg` is on your PATH.

Key runtime libraries:
- UI: `PySide6`, `pyvista`, `pyvistaqt`, `vtk`
- Geometry: `numpy`, `scipy`, `trimesh`, `networkx`
- GLB: `pygltflib`
- Rendering: `moderngl`, `PyOpenGL` (legacy/optional)

## Module Relationship Overview

Core layers and responsibilities:
- `rigging/` (data + algorithms)
  - `mesh_io.py`: mesh I/O, validation, and preprocessing.
  - `gltf_loader.py`: loads GLB mesh + skeleton + bind matrices.
  - `gltf_export.py`: exports skeleton-only GLB.
  - `skeleton.py`: joint hierarchy, bind locals/globals, FK, skinning matrices.
  - `lbs.py`: Linear Blend Skinning (LBS) on CPU.
  - `weights_heat.py`: heat diffusion weights (Pinocchio-style).
  - `weights_nearest.py`: nearest-joint weights baseline.
  - `rig_io.py`: rig bundle save/load (NPZ).
  - `skeleton_optimize.py`: quadruped pose cleanup helpers.
  - `constraints.py`: minimal IK/constraints scaffold (Fabrik).
- `scene/` (animation data model)
  - `asset.py`: mesh + skeleton + weights container.
  - `timeline.py`: keyframe timeline and interpolation.
  - `actions.py`: procedural clips and blending.
  - `choreography.py`: multi-asset scheduling.
- `ui/` (interactive editor)
  - `skeleton_editor.py`: main app controller, state, and data pipeline.
  - `viewport.py`: PyVista/VTK viewport, picking, gizmos, model transform.
  - `action_timeline.py`: keyframe record/playback UI.
  - `auto_motion.py`: auto head-shake / walking keyframe generators.
  - `export_panel.py`: load/export controls and render settings.
  - `compute_worker.py`: worker threads for deform + weight compute.
  - `weight_tools.py`: weight compute helpers for UI.
  - `style.py`: UI theme.
  - `app.py` / `ui_simple.py`: UI entrypoints.
- `render/` (render backends)
  - `offscreen_vtk.py`: offscreen VTK rendering for frame export.
  - `offscreen_mgl.py`: moderngl offscreen rendering (legacy/optional).
  - `onscreen_qt.py`: onscreen bridge (legacy/optional).
- `tools/` (headless utilities)
  - `bake_frames.py`: batch frame rendering without UI.
  - `export_video.py`: frames -> video via ffmpeg.
  - `preview_obj.py`: quick mesh preview.
- `examples/`
  - `head_shake_demo.py`: simple keyframe demo scene.

Detailed data flow:
1) `ui/skeleton_editor.py` calls `rigging/gltf_loader.py` to load mesh, joints, and bind matrices.
2) `rigging/skeleton.py` builds bind locals/globals; `rigging/lbs.py` prepares skinning matrices.
3) Weights are computed by `rigging/weights_heat.py` (default) or `rigging/weights_nearest.py` (fallback).
4) `ui/compute_worker.py` runs deformation/weight tasks off the UI thread, returning deformed vertices.
5) `ui/action_timeline.py` records/plays keyframes; `ui/auto_motion.py` can auto-generate clips.
6) `ui/viewport.py` renders the current mesh + skeleton; picking/gizmos update joint transforms.
7) `render/offscreen_vtk.py` renders frames to disk when exporting; `tools/export_video.py` encodes videos.

Class dependency map (key classes):
- `Mesh` (`rigging/mesh_io.py`) is the base geometry type used by `RiggedAsset` and `SpotRigWindow`.
- `Joint` and `Skeleton` (`rigging/skeleton.py`) define the hierarchy and FK; used by `SpotRigWindow`,
  `RiggedAsset`, `DeformWorker`, and `WeightsWorker`.
- `HeatWeightsConfig` (`rigging/weights_heat.py`) configures heat weights; passed into `WeightsWorker`
  from `SpotRigWindow`.
- `TopKWeights` (`rigging/lbs.py`) is a compact weight format used by `SpotRigWindow` and workers.
- `RiggedAsset` (`scene/asset.py`) aggregates `Mesh`, `Skeleton`, and weights; `Scene` simulates it.
- `Keyframe` -> `Track` -> `Timeline` (`scene/timeline.py`) form the core animation container.
- `ActionClip` depends on `Timeline`; `ActionState` depends on `ActionClip` (`scene/actions.py`).
- `Scene` depends on `RiggedAsset` and `Timeline`; returns `AssetFrame` (`scene/choreography.py`).
- `SpotRigWindow` (`ui/skeleton_editor.py`) is the app controller; it owns `RigControlPanel`,
  `ActionTimeline`, `RigViewport`, `DeformWorker`, and `WeightsWorker`.
- `RigControlPanel` (`ui/export_panel.py`) embeds `SkeletonCompilePanel` and forwards UI callbacks
  to `SpotRigWindow`.
- `ActionTimeline` (`ui/action_timeline.py`) drives keyframes via `SpotRigWindow` callbacks.
- `RigViewport` (`ui/viewport.py`) pulls state from `SpotRigWindow` to render and push interaction edits.
- `OffscreenVtkRenderer` (`render/offscreen_vtk.py`) is used by `SpotRigWindow.render_frames_vtk`.

## Main Runtime Flow (UI)

1) Load a GLB model (mesh + skeleton) in the UI.
2) (Optional) Enable compile mode to edit joints without deforming the mesh.
3) Recompute weights (heat or nearest-joint).
4) Edit joints / record keyframes in the timeline.
5) Use Auto Head Shake / Auto Walking to generate demo keyframes.
6) Render frames (VTK) to `out/frames/...`.
7) Make video (ffmpeg) to `out/videos/...`.

## How to Run (Demos)

### UI (recommended)
```bash
python -m ui.ui_simple
```

Typical UI demo steps:
1) Model -> Browse -> Load
2) (Optional) Enable "Skeleton compile mode" and edit joints
3) "Recompute weights"
4) "Auto head shake" or "Auto walking"
5) "Render frames (VTK)"
6) "Make video"

### Headless frame baking
```bash
python -m tools.bake_frames \
  --scene-module examples.head_shake_demo \
  --scene-func build_scene \
  --out out/frames/head_shake \
  --duration 1.5 \
  --fps 30
```

Then encode video:
```bash
python -m tools.export_video \
  --frames out/frames/head_shake \
  --video out/videos/head_shake.mp4 \
  --fps 30
```

## Feature Demos

- Skeleton compile mode:
  - Toggle "Skeleton compile mode"
  - Add joints / set parent / clear parent
  - Restore original skeleton
- Weight computation:
  - Use "Recompute weights"
  - Switch skinning mode (full vs nearest-joint)
- Auto motions:
  - "Auto head shake" generates keyframes
  - "Auto walking" generates a longer looped clip
- Rendering:
  - "Render frames (VTK)" writes PNGs
  - "Make video" uses ffmpeg to produce MP4
- Debug exports:
  - Export joint positions, edges, and bind weights
