# 项目中期文档

## 1. 骨架关节点的创建思路

### 1.1 目标与约束

在本项目需要实现的系统里，**骨架**（skeleton）是驱动网格形变的抽象控制结构，设计时要同时满足以下要素：

- 拓扑上，能覆盖角色主要的运动 DOF：躯干、四肢、头颈等；

- 几何上，关节点位置要与网格形状匹配：大致位于体积中心线或关节转折处；

- 数值上，骨骼深度和父子层级要适合后续的 FK、Heat Weights 与 LBS 计算。

### 1.2 骨架结构设计

中期阶段项目主要分析的模型是`spot`，对于这类模型，项目暂时实现了通用的四足动物骨骼范式：

|部位|对应骨骼|作用|
|---|---|---|
|躯干主干链（root/body）|`body_top0 -> body_top1 -> ...`|对应动物背部的中心线；核心作用是整体位移、弯曲和躯干扭转。|
|颈部 + 头部链（neck/head）|从躯干上部某个关节出发：`neck0 -> neck1 -> head0 -> head0_end`|用于控制抬头、点头、左右转头；末端点可作为看向目标的 IK 终端。|
|前腿链（left/right front leg）|每条腿一般 3–4 段：`leg_front_left_top0 -> leg_front_left_top1 -> leg_front_left_bot0 -> ...`|从躯干下部沿法线方向向下，经过肩、肘、腕等动作位置。|
|后腿链（left/right hind leg）|类似前腿：`leg_hind_left_top0 -> ... -> leg_hind_left_bot2_end`|摆放位置略偏后；用于控制跳跃、行走、蹲伏等动作。
|
|尾巴 / 耳朵等附属链||用单独的分支关节链表示，后续用于增加动画张力以及用于多模型交互，但不影响主躯干权重分布。|


![](./static/skeleton_nodes.png)

在具体实现上，本项目进行了两层抽象：

- 数据层：

    -  使用 `Skeleton / Joint` 类描述：

    - `Joint.name`：唯一标识

    - `Joint.parent`：父节点 index

    - `Joint.bind_local`：在 Bind Pose 下的局部变换矩阵（相对父节点）

    -  支持从 GLB/FBX 的 skins + inverseBindMatrices 中自动构建。

- 计算层：

    - 骨架类提供 Forward Kinematics：

    - 输入每个关节的局部姿态（旋转/平移），输出 global_mats（世界变换）；

    - 用于 LBS、Gizmo 显示和 UI 交互。

### 与 GLB 骨架的对应关系

1. 当前 pipeline 中，`Spot` 模型的骨架是在 **DCC 工具**（Blender）中手动完成，以 GLB 形式导出。

2. 得到 GLB 文件后，项目在 `gltf_loader.py` 中实现了骨骼加载器，功能包括：读取 mesh（顶点/面）；读取 `skin.joints` 列表，按 parent 关系恢复关节树；使用 inverseBindMatrices 反推 Bind Pose 下的 `global_mats`；并由 `global_mats` 和 parent 关系推回 `bind_local`，构造 `Skeleton` 实例。

3. 在经过以上步骤后，基本可以保证  骨架结构与 Blender 中完全一致，如果重新在 DCC 里调整骨架，只要重新导出 GLB 即可复用。

![](./static/loading.png)

## 2. 模型表面点权重的算法设计

### 2.1 最近骨骼 / 双骨插值（weights_nearest.py）

这是一个基础、快速的权重初始化方法，在项目中用作测试，以及 Heat 权重的 warm start 或 fallback。该算法的核心思路如下：

1. **距离度量**：

   - 把每根骨骼看作线段（父关节 (p)、子关节 (c)）；
   - 对每个顶点 (v)，计算到若干骨骼线段的**最短距离** (d_k)；
   - 只保留距离最近的前 (K) 根骨骼（比如 2–4 根）。

2. **权重分配**：

   - 将距离反比转为权重：
     $$[\tilde{w}_k = \frac{1}{d_k^\alpha + \epsilon},\quad w_k = \frac{\tilde{w}_k}{\sum_j \tilde{w}_j}]$$
   - 通常取$(\alpha \in [1,2])$，调整衰减速度。
3. **拓扑一致性优化（双线性/双骨插值）**：

   - 对于位于肢体中部的顶点 (v)，会同时靠近父骨和子骨；
   - 可以基于顶点在骨骼方向上的投影参数 ($t\in[0,1]$) 做线性插值，调节父子骨权重比例；
   - 这样关节弯曲时形变更自然。

双骨差值法仅依赖少量几何操作（点到线段距离），实现简单；但是这种蒙皮方式对骨骼密度和布置较敏感，在复杂拓扑处容易出现权重不连续。

双骨插值的部分关键代码是：

```python
def compute_nearest_bilinear_weights(
    verts: np.ndarray,            # (N,3)
    skel,                         # rigging.skeleton.Skeleton
    config: Optional[NearestBilinearConfig] = None,
) -> np.ndarray:
    if config is None:
        config = NearestBilinearConfig()

    V = np.asarray(verts, dtype=np.float32)
    N = V.shape[0]
    parents = skel.parents()
    edges = _bone_edges_from_parents(parents)    # (B,2)
    if edges.size == 0:
        raise ValueError("Skeleton has no bones (no parent-child pairs).")

    # Bind joint positions and bone endpoints
    J_pos = _global_bind_positions(skel)        # (J,3)
    A = J_pos[edges[:, 0]]                      # (B,3) parent
    B = J_pos[edges[:, 1]]                      # (B,3) child
    AB = B - A
    seg_lengths = np.linalg.norm(AB, axis=1)    # (B,)

    sigma = config.sigma if (config.sigma is not None) else _auto_sigma(seg_lengths)

    # Prepare output
    J = J_pos.shape[0]
    W = np.zeros((N, J), dtype=np.float32)

    # Chunked processing to limit memory footprint
    bs = int(config.chunk_size)
    for start in range(0, N, bs):
        end = min(N, start + bs)
        Vc = V[start:end]  # (Nv,3)

        t, d, _ab2 = _project_points_to_segments_batch(Vc, A, B)  # (Nv,B) each

        # pick k nearest bones per vertex
        K = int(max(1, config.k_bones))
        if K >= t.shape[1]:
            K = t.shape[1]

        # indices of K smallest distances along axis=1
        # argpartition faster than argsort
        idx_part = np.argpartition(d, K-1, axis=1)[:, :K]           # (Nv,K)
        # sort those K by distance ascending for stability
        row_ids = np.arange(idx_part.shape[0])[:, None]
        d_sorted = np.take_along_axis(d, idx_part, axis=1)
        order = np.argsort(d_sorted, axis=1)
        bone_ids = np.take_along_axis(idx_part, order, axis=1)      # (Nv,K)
        d_sorted = np.take_along_axis(d_sorted, order, axis=1)
        t_sorted = np.take_along_axis(t, idx_part, axis=1)
        t_sorted = np.take_along_axis(t_sorted, order, axis=1)

        # radial falloff per bone
        F = _radial_weight(d_sorted, sigma=sigma, mode=config.falloff, eps=config.eps)  # (Nv,K)

        # accumulate contributions to joints
        for k in range(K):
            b_ids = bone_ids[:, k]                   # (Nv,)
            t_k = t_sorted[:, k]                     # (Nv,)
            f_k = F[:, k]                            # (Nv,)

            j_parent = edges[b_ids, 0]               # (Nv,)
            j_child  = edges[b_ids, 1]               # (Nv,)

            w_parent = (1.0 - t_k) * f_k
            w_child  = t_k * f_k

            # scatter-add
            W[start:end, j_parent] += w_parent
            W[start:end, j_child]  += w_child

        # prune to max_influences if requested (dense path)
        if config.max_influences and config.max_influences > 0:
            mi = int(config.max_influences)
            # keep top-mi per row
            # argpartition descending
            # NOTE: we operate on the chunk only
            idx_desc = np.argpartition(-W[start:end], kth=mi-1, axis=1)
            keep_cols = idx_desc[:, :mi]
            mask = np.zeros_like(W[start:end], dtype=bool)
            rr = np.arange(end - start)[:, None]
            mask[rr, keep_cols] = True
            W[start:end][~mask] = 0.0

        # renormalize chunk
        if config.renormalize:
            rowsum = np.sum(W[start:end], axis=1, keepdims=True)
            rowsum[rowsum < config.eps] = 1.0
            W[start:end] /= rowsum

    return W

```

### 2.2 Heat Diffusion 权重（weights_heat.py）

这是主要的权重算法，参考 Pinocchio 等工作中的**基于热传导的骨骼权重**思想。核心思想是：

> 把每个骨骼关节当作“热源”，在三角形网格上做热扩散，最终稳态温度场即为该关节的权重分布。

#### 2.2.1 离散拉普拉斯与热方程

在网格上，项目构造了**离散拉普拉斯矩阵** ($L \in \mathbb{R}^{n\times n}$)（(n) 为顶点数），例如使用 cotangent 权重：

$$[
(Lu)*i = \sum*{j \in N(i)} w_{ij}(u_i - u_j)
]$$

对每个关节 (j)，希望求出权重函数 ($w^{(j)}\in\mathbb{R}^n$)，满足离散热方程稳态形式：

$$[
(L + \tau I), w^{(j)} = b^{(j)}
]$$

其中：

- ($\tau > 0$) 为热扩散系数（控制平滑程度）；
- ($b^{(j)}$) 是“热源项”：在靠近关节的若干顶点上赋 1，其余为 0。

**直观理解**：

- (L) 惩罚空间梯度（鼓励平滑）；
- ($\tau I$) 把权重“拉回”热源；
- 解 ($w^{(j)}$) 相当于**在网格上做一次带衰减的散热**。

#### 2.2.2 线性系统与数值求解

对每个关节 (j) 解一个稀疏线性系统：

$$[
(L + \tau I), w^{(j)} = b^{(j)}
]$$

数值实现上：

- 预先用 `scipy.sparse` 构建好 (A = L + \tau I)；
- 对各关节频繁调用 `cg(A, b_j, ...)`（共轭梯度法）；
- 为了避免共轭梯度不收敛，通常会：

  - 选取适度的 (\tau)；
  - 对 `cg` 设置合理的 `maxiter` 和残差阈值；
  - 失败时 fallback 到最近骨骼权重。

最终把所有关节的结果堆叠，得到权重矩阵 ($W \in \mathbb{R}^{n \times J}$)，再对每个顶点的权重 `W[i]` 做归一化：

$$[
W_{ij} \leftarrow \frac{\max(W_{ij}, 0)}{\sum_k \max(W_{ik}, 0) + \varepsilon}
]$$

#### 2.2.3 与 LBS 的结合

一旦有了 (W) 和 FK 算出的关节矩阵 (T_j)，就可以进行标准的**线性混合蒙皮**：

$$[
v_i' = \sum_j W_{ij} ; (T_j , T_j^{\text{bind}^{-1}}) ; \hat{v}_i
]$$

其中：

- ($\hat{v}_i$) 为 Bind Pose 下的齐次坐标；
- ($T_j^{\text{bind}}$) 由 GLB 的 `inverseBindMatrices` 或 Skeleton 的绑定位姿推得。

Heat 权重的好处包括以下要点：

- 权重场在模型表面**连续且平滑**；
- 在关节附近自然实现多骨“混合”控制，弯曲时不易产生明显折痕；
- 对骨架布置有一定鲁棒性，不要求手动刷权。

Heat权重的关键算法内容包括：

```python
    # σ 自动估计：中位骨段长度的一半（经验值）
    if cfg.sigma is None:
        med = float(np.median(seg_len))
        cfg_sigma = max(0.5 * med, 1e-3)
    else:
        cfg_sigma = float(cfg.sigma)

    # 计算网格邻接与拉普拉斯
    neighbors, L = compute_vertex_adjacency(mesh)
    if L is None:
        raise RuntimeError("No Laplacian available (SciPy missing?).")

    # 随机游走标准化 Laplacian：L_rw = D^{-1} L
    deg = np.asarray(L.sum(axis=1)).ravel()
    deg[deg < 1e-16] = 1.0
    Dinv = sp.diags(1.0 / deg, 0, shape=L.shape)
    L_rw = Dinv @ L

    # A = I - tau * L_rw
    I = sp.identity(L.shape[0], format="csr", dtype=np.float32)
    A_sys = (I - cfg.tau * L_rw).tocsr()

    # 预处理/分解（一次，多 RHS 复用）
    solver_mode = cfg.solver.lower()
    do_splu = (solver_mode == "splu" or solver_mode == "auto")
    lu = None
    if do_splu:
        try:
            lu = spla.splu(A_sys.tocsc())
        except Exception:
            # 回退到 CG
            lu = None
            solver_mode = "cg"
```

### 2.3 蒙皮效果展示

尝试给 `Spot` 模型添加一个“向前走路”的动作：

- 最近骨骼权重：

![](./static/nearest.png)

- Heat权重：

![](./static/heat.png)

## 3. 参考文献列表

|名称|文献链接|
|---|---|
|Ilya Baran and Jovan Popović. 2007. Automatic rigging and animation of 3D characters. ACM Trans. Graph. 26, 3 (July 2007), 72–es.|https://www.cs.toronto.edu/~jacobson/seminar/baran-and-popovic-2007.pdf|
|Automatic Skinning using the Mixed Finite Element Method (Arxiv)|https://arxiv.org/html/2408.04066v1|











