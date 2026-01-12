from examples.head_shake_demo import build_scene
scene, _ = build_scene()
asset = scene.assets[0]
names = [j.name for j in asset.skeleton.joints]
head = "head0"
j = names.index(head)
print("head0 max weight:", asset.weights[:, j].max())
print("head0 nonzero:", (asset.weights[:, j] > 1e-6).sum())
v0 = scene.simulate(0.0)[0].vertices
v1 = scene.simulate(0.5)[0].vertices
print("max delta:", (abs(v1 - v0)).max())
