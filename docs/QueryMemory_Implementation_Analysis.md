# Spatio-Temporally Aligned Confidence-Gated Causal Query Memory
# 实现分析报告

> 基于 SparseWorld 和 OccWorld 代码的深度分析
> 生成日期: 2026-08-02

---

## A. 原始代码调用链

### A.1 完整调用链

```
NuScenesDatasetOccpancy4DTraj                          [dataset]
  ├── PrepareImageInputs                               多视角图像加载+增强
  ├── LoadOccGTFromFile4DTraj                           当前帧+6未来帧占用GT
  ├── LoadAnnotationsBEVDepth4DTraj                     BDA数据增强（翻转）
  └── DefaultFormatBundle3D                             格式化
         │
         ▼
SparseWorld4DTraj.forward_train()                      [顶层检测器]
  │
  ├── extract_feat(img, img_metas)                     [OPUS基类]
  │     ├── GridMask / GpuPhotoMetricDistortion        数据增强
  │     ├── img_backbone (ResNet-50)                   图像特征提取
  │     └── img_neck (FPN)                             多尺度特征金字塔
  │           → mlvl_feats: list of [B, T*N, G*C, H, W]
  │
  ├── pts_bbox_head.forward(mlvl_feats, img_metas)     [OPUSHead] ═══ RAP 入口
  │     ├── init_points.weight → [B, 1040, 1, 3]       可学习query初始化(归一化坐标)
  │     ├── points_scale → 动态缩放 [0.8, 1.5]          由ego状态控制
  │     ├── query_feat = zeros(B, 1040, 256)            初始特征为零向量
  │     ├── get_matched_inds(num_stamps_all)             ═══ TASS: 分配query到时间步
  │     │     → ind_stamps_all: [1040], 值∈{0,1,2,3,4,5,6}
  │     ├── reset_mask()                                因果mask生成 [1040, 1040]
  │     │
  │     └── OPUSTransformer.forward()                   ═══ RAP Transformer
  │           └── OPUSTransformerDecoder.forward()
  │                 ├── occ2img = lidar2img @ ego2lidar  投影矩阵构建
  │                 └── for layer in 6 decoder layers:   逐层精炼
  │                       ├── query_points = query_points.detach()  截断梯度
  │                       └── OPUSTransformerDecoderLayer.forward()
  │                             ├── position_encoder(query_points)   位置编码
  │                             ├── OPUSSampling → sampling_4d       3D→2D采样
  │                             ├── AdaptiveMixing                   通道/点混合
  │                             ├── OPUSSelfAttention + ind_mask     距离自适应注意力+因果约束
  │                             ├── FFN                              前馈网络
  │                             ├── cls_branch → [B,1040,P,17]      分类
  │                             ├── reg_branch → [B,1040,P,3]       回归
  │                             └── refine_points                    点坐标精炼
  │
  │     → outs: {query_feat: [B,1040,256],
  │              all_cls_scores: list of 6,
  │              all_refine_pts: list of 6,
  │              init_points: [B,1040,1,3]}
  │
  ├── forward_backbone()                                ═══ SCF 循环入口
  │     ├── plan_head(ego_states) → ego_feat [B,1,256]  ego状态编码
  │     ├── points_scale_branch(ego_feat)               动态query缩放
  │     │
  │     ├── 分离当前/未来 query:
  │     │   curr_query_feat = query_feat[:, ind_stamps_all==0]  → [B, 720, 256]
  │     │   curr_query_pos  = query_pos[:, ind_stamps_all==0]   → [B, 720, 48, 3]
  │     │   curr_query_cls  = query_cls[:, ind_stamps_all==0]   → [B, 720, 48, 17]
  │     │
  │     │   ★★★ 推荐记忆读取插入点 ★★★
  │     │
  │     └── for interval in range(num_fu_frames):       ═══ SCF 自回归循环
  │           ├── ego_cross_attn(ego_feat, curr_query)  ego⊗场景交叉注意力
  │           │     → fused_ego_feat [B, 1, 256]
  │           ├── traj_head(fused_ego_feat)             → pred_traj [B, 1, 2]
  │           ├── 拼接未来query:
  │           │   curr_query_feat ← cat(curr, query_feat[:, ind==interval+1])
  │           │   curr_query_pos  ← cat(curr, query_pos[:, ind==interval+1])
  │           ├── curr_query_feat += fused_ego_feat + position_encoder(pos+timestamp)
  │           ├── reg_branch → reg_offset [B, Q', 48, 3]
  │           ├── cls_branch → cls_score  [B, Q', 48, 17]
  │           ├── vel_branch → vel_offset [B, Q', 48, 2]
  │           ├── 运动物体(class 2-10): reg_offset[:,:2] += vel_offset * moving_mask
  │           ├── refine_points(curr_query_pos, reg_offset) → 更新位置
  │           └── [训练] trans_points + loss_future
  │
  ├── loss_pretrain(voxel_semantics, temporal_semantics, temporal2ego, outs)
  │     ├── get_sparse_voxels(voxel_semantics)          稠密GT→稀疏 (当前帧)
  │     ├── get_sparse_voxels_stack(temporal_semantics, temporal2ego)
  │     │     → 多帧GT对齐到当前ego坐标系, 堆叠到400×300×24网格
  │     ├── _get_target_single (KNN最近邻匹配)
  │     └── loss_single_mask / loss_single_rangemask    cls_loss + pts_loss
  │
  ├── loss_future(forecast_points, forecast_semantics, temporal_semantics)
  │     → 各未来帧的分类+回归损失
  │
  └── loss_traj(pred_trajs, gt_trajs)
        → L2轨迹损失 (1s/2s/3s)
```

### A.2 关键发现

1. **模型没有跨 forward pass 的时序记忆。** 每次调用 `pts_bbox_head.forward()` 都从零向量 `query_feat` 和固定 `init_points` 重新开始。`self.memory` 仅缓存图像特征（用于在线推理避免重复提取backbone特征），不缓存 query 状态。

2. **因果约束仅存在于单次前向中。** `ind_mask` 是 [1040, 1040] 的静态 mask，防止时间步 t 的 query 关注时间步 > t 的 query。它在 RAP Transformer 的 6 层 self-attention 中使用，但不涉及跨帧的历史信息。

3. **SCF 循环中没有任何形式的历史 query 缓存或读取。** 每一步仅接收 ego 特征 + 当前 query 集合，没有外部记忆输入。

4. **所有 query 坐标都在当前 ego 坐标系中。** 通过 `encode_points`/`decode_points` 在归一化 [0,1] 和物理米制之间转换。

---

## B. 关键张量表

### B.1 Query 特征与位置

| 张量 | 代码变量 | 形状 | 坐标系/物理含义 | 产生位置 | 消费位置 |
|------|----------|------|----------------|----------|----------|
| Query 初始位置 | `init_points.weight` | `[1040, 3]` | 归一化 [0,1] ego坐标 | `OPUSHead.__init__` (opus_head.py:92) | `OPUSHead.forward` (opus_head.py:103) |
| 缩放后初始位置 | `init_points` | `[B, 1040, 1, 3]` | 归一化, 乘以 points_scale | `OPUSHead.forward` (opus_head.py:105) | `OPUSTransformer` 输入 |
| Query 特征(初始) | `query_feat` | `[B, 1040, 256]` | 全零向量 | `OPUSHead.forward` (opus_head.py:109) | `OPUSTransformer` 输入 |
| Query 特征(RAP后) | `query_feat` / `outs['query_feat']` | `[B, 1040, 256]` | 语义特征 | `OPUSTransformer` 输出 | SCF 循环 |
| 精炼点位置(RAP后) | `outs['all_refine_pts'][-1]` | `[B, 1040, 48, 3]` | 归一化 [0,1] ego坐标 | 第6层 `refine_points` | SCF 循环 |
| 当前帧 query 特征 | `curr_query_feat` | `[B, 720, 256]` → 逐步增长至 `[B, 1040, 256]` | 语义特征 | `forward_backbone` (sw4d:244) | SCF 循环各步 |
| 当前帧 query 位置 | `curr_query_pos` | `[B, 720, 48, 3]` → 逐步增长 | 归一化 [0,1] ego坐标 | `forward_backbone` (sw4d:245) | SCF 循环各步 |

### B.2 时间标签与调度

| 张量 | 代码变量 | 形状 | 物理含义 | 产生位置 | 消费位置 |
|------|----------|------|---------|----------|----------|
| Query 时间分配 | `ind_stamps_all` | `[1040]` | 每个query所属时间步 (0-6) | `get_matched_inds` (bbox/utils.py:102) | `forward_backbone` 中分离query |
| 时间统计缓冲区 | `num_stamps_all` | `[1040, 7]` | 每个query在各时间步的GT近邻统计 | `OPUSHead.__init__` (opus_head.py:88) | `get_matched_inds` 输入 |
| 因果 mask | `ind_mask` | `[1040, 1040]` | 0=允许, -1e5=屏蔽 | `reset_mask` (opus_head.py:156) | `OPUSSelfAttention` (transformer.py:365) |
| Query 时间戳 | `curr_query_timestamp` | `[B, Q', 48, 1]` | 0.0(当前) or 0.5(未来) | `forward_backbone` (sw4d:248,270) | `position_encoder` 输入 |

### B.3 Ego 状态与位姿

| 张量 | 代码变量 | 形状 | 物理含义 | 产生位置 | 消费位置 |
|------|----------|------|---------|----------|----------|
| Ego 运动状态 | `temporal_ego_states` | `dict{0-5: [1, 21]}` | 速度/加速度等运动学特征 | 数据集 (dataset:453) | `forward_backbone` |
| Ego 特征 | `ego_feat` | `[B, 1, 256]` | 编码后的ego状态 | `plan_head` (sw4d:228) | `ego_cross_attn`, `position_encoder` |
| Ego2Global | `ego2global` | `[B, 4, 4]` | 当前帧ego→全局坐标变换 | 数据集 (nuscenes_dataset.py:239) | `trans_coords` |
| Temporal2Ego | `temporal2ego` | `dict{0-5: [4, 4]}` | `inv(curr_ego2global) @ fut_ego2global` | 数据集 (dataset:482) / `trans_coords` | `loss_pretrain`, `get_sparse_voxels_stack` |
| Ego 轨迹 GT | `temporal_trajs` | `[B, 6, 2]` | 未来6步(3s)的2D位移 | 数据集 | `loss_traj` |

### B.4 分类/回归输出（置信度来源）

| 张量 | 代码变量 | 形状 | 物理含义 | 产生位置 | 消费位置 |
|------|----------|------|---------|----------|----------|
| RAP 分类分数 | `all_cls_scores[-1]` | `[B, 1040, 48, 17]` | Raw logits (Focal Loss用) | Transformer Layer 6 cls_branch | `get_occ` (.sigmoid()) |
| SCF 分类分数 | `cls_score` | `[B, Q', 48, 17]` | Raw logits, SCF每步更新 | `self.cls_branch` (sw4d:278) | 运动物体判断, loss_future |
| 回归偏移 | `reg_offset` | `[B, Q', 48, 3]` | 位置偏移(归一化空间) | `self.reg_branch` (sw4d:277) | `refine_points` |
| 速度偏移 | `vel_offset` | `[B, Q', 48, 2]` | XY方向速度 | `self.vel_branch` (sw4d:279) | 运动物体位置更新 |

### B.5 投影矩阵

| 张量 | 代码变量 | 形状 | 物理含义 | 产生位置 |
|------|----------|------|---------|----------|
| Occ→Image | `occ2img` | `[B, N, 4, 4]` | ego坐标→像素坐标 (`lidar2img @ ego2lidar`) | `OPUSTransformerDecoder.forward` (transformer.py:115) |
| Ego→Lidar | `ego2lidar` | `[B, 4, 4]` | ego坐标→lidar坐标 | 数据集 `img_metas` |
| Lidar→Image | `lidar2img` | `[B, N, 4, 4]` | lidar坐标→各相机像素 | 数据集 `img_metas` |

---

## C. OccWorld 与 SparseWorld 的代码映射

### C.1 功能对应表

| 功能 | OccWorld 实现 | SparseWorld 现状 | 新模块承担 |
|------|-------------|-----------------|-----------|
| **历史状态组织** | `TransVQVAE`: 16帧VQ token序列 `[B,16,128,50,50]`, 全部一次性输入Transformer | **不存在**。每次从零开始，无持久状态 | `QueryMemoryBank`: 保存K帧历史query特征+位置 |
| **因果读取** | `PlanUAutoRegTransformer.attn_mask`: `[15,15]` 下三角mask，每帧只看过去帧 | `OPUSHead.reset_mask()`: `[1040,1040]` ind_mask，当前步query不看未来步query（单次前向内部） | `CausalQueryMemoryAttention`: 跨帧因果读取历史memory |
| **自回归预测** | `forward_autoreg_with_pose`: 逐帧预测→argmax→VQ lookup→拼接→继续 | SCF循环: 逐步拼接未来query→ego融合→reg/cls/vel→refine | 保持SCF不变，仅在循环前/中注入记忆 |
| **Ego pose编码** | `PoseEncoder`: MLP `Linear(5,128)→ReLU→Linear(128,128)`，2D位移+3-mode one-hot | `plan_head`: MLP `Linear(21,256)→ReLU→…→Linear(256,256)`，3D速度×7帧 | `EgoPoseAligner`: 用4×4变换矩阵做显式坐标对齐 |
| **Pose↔Scene交互** | Dual-stream: pose query ⊗ scene token cross-attention (每U-Net层) | `ego_cross_attn`: ego单token对所有scene query的交叉注意力 (SCF每步) | 复用现有 `ego_cross_attn`，不需额外pose-scene交互 |
| **Teacher forcing** | 训练全程使用GT token + causal mask | SCF训练时用GT occupation做loss，但预测过程本身是自回归的（无GT注入中间步） | 保持现有训练方式 |

### C.2 OccWorld 可借鉴的设计思想

1. **因果 mask 的优雅设计**：OccWorld 的 `[15,15]` 下三角 mask 简洁地实现了时序因果约束。在 Memory Attention 中，我们需要类似的 mask：当前帧 query 只能读取 ≤ 当前时间步的 memory。

2. **Pose token 与 scene token 的双流设计**：OccWorld 让 pose 和 scene 在两个平行流中交互。我们的设计中，ego pose 信息用于坐标对齐（几何层面）而非仅作为特征辅助（OccWorld方式），这是一个更强的归纳偏置。

3. **Temporal positional embedding**：OccWorld 对不同时间步使用不同的 learned temporal embedding。我们的 Memory Attention 中的 λ_t Δt 项起到类似作用，但以连续时间差代替离散位置编码。

### C.3 不可直接复用的 OccWorld 代码

1. **U-Net 结构的 Conv2D 空间处理**：依赖固定 50×50 网格，不适用于动态稀疏 query。
2. **VQ-VAE codebook 预测**：OccWorld 预测 512-way codebook index（离散输出）。SparseWorld 直接预测连续的位置和语义 logits。
3. **`(b*h*w, f, c)` reshape 的时序注意力**：固定空间网格假设。对应到稀疏 query 应为 `(b*Q, K_mem, c)`，但我们使用显式跨帧注意力而非此种 reshape。
4. **autoregressive 逐帧重跑整个 Transformer**：OccWorld 没有 KV-cache，每步重跑全部。SparseWorld 的 SCF 循环更高效（仅MLP分支，不重跑Transformer）。

---

## D. 推荐接入点

### D.1 方案对比

#### 方案 1：RAP 与 SCF 之间单次读取观测记忆

```
pts_bbox_head.forward() → RAP 输出 query_feat, query_pos
                                    ↓
                        ★ Memory Read (单次) ★
                        curr_query_feat = MemoryAttention(curr_query, memory)
                                    ↓
                        SCF autoregressive loop (6 步)
```

| 维度 | 分析 |
|------|------|
| **改动量** | 最小。仅需修改 `forward_backbone()` 中约 15 行，加入 memory read |
| **训练稳定性** | 高。memory read 在 Transformer 之后、SCF 之前，梯度路径清晰。历史 memory 可 detach 避免跨时间图 |
| **是否缓解递归误差** | 部分。增强了 SCF 循环的初始输入质量，但 SCF 内部的递归误差累积未直接干预 |
| **显存/速度** | 极小开销。720 queries × K×720 memory 的注意力矩阵约 10-80MB（K=1-10），远小于 RAP Transformer 本身 |
| **与 TS-MHSA 功能重复** | 无。TS-MHSA (ind_mask) 仅约束单次前向内不同时间步 query 间的注意力；Memory Attention 引入的是来自前几帧 forward pass 的历史信息，完全正交 |

#### 方案 2：SCF 每步递归预测前读取观测 + 预测记忆

```
pts_bbox_head.forward() → RAP 输出
                                    ↓
for interval in range(6):
    ★ Memory Read (每步) ★
    curr_query_feat = MemoryAttention(curr_query, obs_memory + pred_memory)
    ego_cross_attn → traj_head
    cat future queries → reg/cls/vel → refine
    ★ Memory Write: 预测 query → pred_memory ★
```

| 维度 | 分析 |
|------|------|
| **改动量** | 中等。需要在 SCF 循环内加入 read + write，并管理 pred_memory 的动态增长 |
| **训练稳定性** | 较低。每步的 memory write/read 形成更长的计算图。pred_memory 需要 detach 否则梯度爆炸。但 detach 后 pred_memory 的梯度信号断裂 |
| **是否缓解递归误差** | 直接缓解。每步都能从观测记忆重新校准，不完全依赖上一步的递归输出 |
| **显存/速度** | 6倍于方案1的 attention 计算。memory 大小随 SCF 步数线性增长（加入预测 query） |
| **与 TS-MHSA 功能重复** | 有一定重叠。SCF 每步的 ego_cross_attn 已经做了 ego↔scene 的交互，再加一次 memory attention 可能导致信息冗余 |

### D.2 推荐方案

**第一版（V1-V4）：方案 1 — RAP 与 SCF 之间单次读取观测记忆。**

理由：
1. **最小侵入性**：仅在 `forward_backbone` 的第 250-260 行（分离 curr_query 之后、SCF 循环之前）插入约 15 行代码。
2. **训练稳定可控**：历史 memory 全部 detach，不引入跨时间步的反向传播。新增参数（Memory Attention 模块）可独立初始化，原权重完全可加载。
3. **逻辑清晰**：观测记忆 = 前几帧 RAP 输出的真实观测 query。不涉及 SCF 预测 query 的写入和质量判断。
4. **效果验证门槛低**：只需确认记忆读取确实提供了有用信息，观察 L2/碰撞率是否下降。
5. **为方案 2 铺路**：V5 阶段可在 V4 基础上自然扩展到 SCF 内部读取，复用已验证的 Memory Attention 模块。

**精确插入位置**：`sparseworld_4d_traj.py`, `forward_backbone()` 方法，在分离 `curr_query_feat` / `curr_query_pos` 之后（约 line 250），SCF `for interval` 循环之前（约 line 261）。

```python
# === 现有代码 (line ~250) ===
curr_query_feat = query_feat[:, ind_stamps_all == 0]        # [B, 720, 256]
curr_query_pos  = query_pos[:, ind_stamps_all == 0].detach() # [B, 720, 48, 3]
curr_query_cls  = query_cls[:, ind_stamps_all == 0]          # [B, 720, 48, 17]

# === 新增: Memory Read ===
if self.memory_enabled and hasattr(self, 'query_memory_bank'):
    curr_query_feat = self.memory_attention(
        curr_query_feat, curr_query_pos, self.query_memory_bank,
        curr_ego2global=img_metas[0]['ego2global']
    )

# === 现有代码继续: SCF loop ===
for interval in range(num_fu_frames):
    ...
```

---

## E. 文件修改清单

### E.1 新增文件

| 文件路径 | 内容 |
|----------|------|
| `mmdet3d/models/sparsedetectors/query_memory.py` | `QueryMemoryBank`, `EgoPoseAligner`, `CausalQueryMemoryAttention`, `ConfidenceGatedFusion` 四个模块 |

### E.2 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py` | 1. `__init__`: 新增 memory 相关子模块的构建（可通过 `memory_enabled` 控制）<br>2. `forward_backbone`: 在 RAP 输出与 SCF 循环之间插入 memory read + gated fusion<br>3. 新增 `_update_memory()` 方法：前向结束后将当前帧 query 写入 memory bank<br>4. 新增 `_clear_memory()` 方法：scene 切换时调用 |
| `mmdet3d/models/sparsedetectors/__init__.py` | 增加 `QueryMemoryBank` 等类的导出 |
| `mmdet3d/models/sparsedetectors/opus.py` | `simple_test_online()`: 在 scene 切换检测处调用 `_clear_memory()` |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py` | 新增 memory 配置参数（见 E.3） |

### E.3 配置参数

```python
# 在 model = dict(...) 中新增:
memory_enabled = True,           # False 时完全恢复原始基线
memory_bank_size = 5,            # 保存最近 K 帧的观测 query
memory_embed_dims = 256,         # memory attention 的特征维度
memory_num_heads = 8,            # 多头注意力头数
memory_dropout = 0.1,            # attention dropout
memory_spatial_radius = 20.0,    # 空间半径 mask (米)，None=不使用
memory_confidence_threshold = 0.3, # 写入 memory 的最低置信度
memory_time_decay_lambda = 0.1,  # 时间衰减系数
memory_pos_penalty_lambda = 0.01, # 距离惩罚系数
memory_conf_bonus_lambda = 0.5,  # 置信度加成系数
```

### E.4 基线恢复机制

设置 `memory_enabled=False` 后：
- `__init__` 中不创建任何 memory 相关模块
- `forward_backbone` 中 memory read 代码被完全跳过（`if self.memory_enabled` 为 False）
- `_update_memory` 和 `_clear_memory` 为空操作
- 模型行为、参数量、计算量与原始 SparseWorld 完全一致
- 原始 checkpoint (`epoch_56.pth`) 可直接加载（新增模块参数不在 checkpoint 中，用 `strict=False`）

---

## F. 模块接口草案

### F.1 QueryMemoryBank

```python
class QueryMemoryBank:
    """
    环形缓冲区，存储最近 K 帧的观测 query 信息。
    每帧存储一条 MemoryEntry。
    """

    def __init__(self, bank_size: int = 5, embed_dims: int = 256,
                 num_refines: int = 48, num_classes: int = 17,
                 confidence_threshold: float = 0.3):
        self.bank_size = bank_size       # K
        self.entries = []                # list of MemoryEntry, 最多 K 条
        # 不继承 nn.Module —— 纯数据容器，无可学习参数

    def write(self,
              query_feat: Tensor,        # [B, Q, C=256]
              query_pos: Tensor,         # [B, Q, 48, 3], 归一化坐标
              cls_scores: Tensor,        # [B, Q, 48, 17], raw logits
              ego2global: Tensor,        # [B, 4, 4]
              timestamp: float,          # 帧时间戳 (秒)
              source_type: str = 'observation',  # 'observation' | 'prediction'
              ) -> None:
        """
        将当前帧 query 写入 memory bank。
        根据置信度阈值过滤低质量 query。
        若已满则淘汰最旧条目。
        """
        confidence = cls_scores.sigmoid().max(dim=-1)[0].mean(dim=-1)  # [B, Q]
        valid_mask = confidence > self.confidence_threshold             # [B, Q]
        entry = MemoryEntry(
            query_feat=query_feat.detach(),
            query_pos=query_pos.detach(),
            confidence=confidence.detach(),
            ego2global=ego2global.detach(),
            timestamp=timestamp,
            source_type=source_type,
            valid_mask=valid_mask.detach(),
        )
        if len(self.entries) >= self.bank_size:
            self.entries.pop(0)          # FIFO 淘汰最旧
        self.entries.append(entry)

    def read_all(self) -> Optional[dict]:
        """
        返回所有 memory 条目的拼接结果。
        Returns:
            None if empty, else dict with:
            - mem_feat:      [B, K*Q_valid, C]
            - mem_pos:       [B, K*Q_valid, 48, 3], 各条目各自ego坐标系
            - mem_confidence:[B, K*Q_valid]
            - mem_ego2global:[K, B, 4, 4], 各条目的ego2global
            - mem_timestamps:[K], 各条目的时间戳
            - mem_valid_mask:[B, K*Q_valid]
            - mem_source_type: list of str, length K
        """

    def clear(self) -> None:
        """scene 切换时调用。"""
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)
```

### F.2 EgoPoseAligner

```python
class EgoPoseAligner(nn.Module):
    """
    将历史 query 位置从其原始 ego 坐标系变换到当前 ego 坐标系。
    """

    def __init__(self, pc_range: list):
        super().__init__()
        self.pc_range = pc_range
        # pc_range = [-40, -40, -1, 40, 40, 5.4]

    def forward(self,
                mem_pos: Tensor,              # [B, M, 48, 3], 归一化坐标
                mem_ego2global: Tensor,        # [B, 4, 4], 历史帧的 ego2global
                curr_ego2global: Tensor,       # [B, 4, 4], 当前帧的 ego2global
                ) -> Tensor:                   # [B, M, 48, 3], 归一化坐标(当前ego系)
        """
        变换步骤:
        1. decode_points: 归一化 → 当前ego坐标系的物理米制
           (注意: mem_pos 存储时是其原始ego系的归一化坐标)
        2. 物理坐标: 原始ego → global → 当前ego
           T = inv(curr_ego2global) @ mem_ego2global
           p_curr = T @ p_hist
        3. encode_points: 物理米制 → 归一化
        """
        # Step 1: 反归一化到物理坐标 (历史 ego 坐标系)
        pos_physical = decode_points(mem_pos, self.pc_range)  # [B, M, 48, 3]

        # Step 2: 构造变换矩阵
        # T_hist2curr = inv(curr_ego2global) @ mem_ego2global
        T = torch.matmul(
            torch.linalg.inv(curr_ego2global),  # [B, 4, 4]
            mem_ego2global                       # [B, 4, 4]
        )  # [B, 4, 4]

        # Step 3: 应用变换
        B, M, P, _ = pos_physical.shape
        pos_flat = pos_physical.reshape(B, M * P, 3)          # [B, M*48, 3]
        pos_transformed = (torch.matmul(pos_flat, T[:, :3, :3].transpose(1, 2))
                          + T[:, None, :3, 3])                  # [B, M*48, 3]
        pos_transformed = pos_transformed.reshape(B, M, P, 3)  # [B, M, 48, 3]

        # Step 4: 重新归一化
        return encode_points(pos_transformed, self.pc_range)    # [B, M, 48, 3]
```

### F.3 CausalQueryMemoryAttention

```python
class CausalQueryMemoryAttention(nn.Module):
    """
    当前 query 对对齐后历史 memory 的软检索注意力。
    注意力分数融合了语义相似度、空间距离惩罚、时间衰减和置信度加成。
    """

    def __init__(self, embed_dims: int = 256, num_heads: int = 8,
                 dropout: float = 0.1,
                 lambda_pos: float = 0.01,
                 lambda_time: float = 0.1,
                 lambda_conf: float = 0.5,
                 spatial_radius: Optional[float] = None,
                 pc_range: list = None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.lambda_pos = lambda_pos
        self.lambda_time = lambda_time
        self.lambda_conf = lambda_conf
        self.spatial_radius = spatial_radius
        self.pc_range = pc_range

        self.W_q = nn.Linear(embed_dims, embed_dims)  # query projection
        self.W_k = nn.Linear(embed_dims, embed_dims)  # key projection
        self.W_v = nn.Linear(embed_dims, embed_dims)  # value projection
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query_feat: Tensor,            # [B, Q, C=256], 当前帧query特征
                query_pos: Tensor,             # [B, Q, 48, 3], 当前帧query位置(归一化)
                mem_feat: Tensor,              # [B, M, C=256], 对齐后的memory特征
                mem_pos_aligned: Tensor,       # [B, M, 48, 3], 对齐到当前ego系的位置(归一化)
                mem_confidence: Tensor,        # [B, M], memory query 置信度
                mem_time_delta: Tensor,        # [B, M], 时间差(秒)
                mem_valid_mask: Tensor,        # [B, M], bool, 有效memory标记
                ) -> Tensor:                   # [B, Q, C], 检索结果 h_i
        """
        A_{ij} = (W_q q_i)^T (W_k m_j) / sqrt(d)
                 - λ_p * |p_i - p̄_j|^2
                 - λ_t * Δt_j
                 + λ_c * log(c_j + ε)

        h_i = Σ_j softmax(A_{ij}) * (W_v m_j)
        """
        B, Q, C = query_feat.shape
        M = mem_feat.shape[1]
        d = C // self.num_heads

        # Semantic similarity: (W_q q) @ (W_k m)^T / sqrt(d)
        q = self.W_q(query_feat)                         # [B, Q, C]
        k = self.W_k(mem_feat)                           # [B, M, C]
        semantic_score = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
                                                          # [B, Q, M]

        # Spatial penalty: -λ_p * |p_i - p̄_j|^2
        q_center = decode_points(query_pos, self.pc_range).mean(dim=2)  # [B, Q, 3]
        m_center = decode_points(mem_pos_aligned, self.pc_range).mean(dim=2)  # [B, M, 3]
        dist_sq = ((q_center.unsqueeze(2) - m_center.unsqueeze(1)) ** 2).sum(-1)
                                                          # [B, Q, M]
        spatial_penalty = -self.lambda_pos * dist_sq

        # Time decay: -λ_t * Δt
        time_penalty = -self.lambda_time * mem_time_delta.unsqueeze(1)
                                                          # [B, 1, M] → broadcast

        # Confidence bonus: +λ_c * log(c + ε)
        conf_bonus = self.lambda_conf * torch.log(mem_confidence + 1e-6).unsqueeze(1)
                                                          # [B, 1, M] → broadcast

        # Combined attention score
        attn = semantic_score + spatial_penalty + time_penalty + conf_bonus
                                                          # [B, Q, M]

        # Spatial radius mask (optional)
        if self.spatial_radius is not None:
            radius_mask = dist_sq > (self.spatial_radius ** 2)
            attn = attn.masked_fill(radius_mask, -1e5)

        # Valid mask
        invalid_mask = ~mem_valid_mask.unsqueeze(1)       # [B, 1, M]
        attn = attn.masked_fill(invalid_mask, -1e5)

        # Softmax + weighted sum
        attn_weights = F.softmax(attn, dim=-1)            # [B, Q, M]
        attn_weights = self.dropout(attn_weights)
        v = self.W_v(mem_feat)                            # [B, M, C]
        h = torch.matmul(attn_weights, v)                 # [B, Q, C]
        h = self.out_proj(h)                               # [B, Q, C]

        return h
```

**效率分析**：
- Q=720, K=5帧 × 720=3600 memory query → 注意力矩阵 720×3600=2.6M entries
- 每个 entry 4 bytes → ~10MB/head, 8 heads → ~80MB, 远小于 GPU 显存
- **全量注意力完全可行**，不需要 top-k 或分帧策略
- 如需进一步优化，`spatial_radius` mask 可将有效 entry 减少到 ~30%

### F.4 ConfidenceGatedFusion

```python
class ConfidenceGatedFusion(nn.Module):
    """
    门控残差融合。历史检索结果通过 learned gate 与当前 query 混合。
    g_i = σ(MLP([q_i, h_i, c_i]))
    q̃_i = q_i + g_i ⊙ FFN(h_i)
    """

    def __init__(self, embed_dims: int = 256, ffn_dims: int = 512):
        super().__init__()
        # Gate: 输入 = [query_feat, memory_output, confidence_embedding]
        # confidence_embedding 来自当前 query 自身的 cls_score
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2 + 1, embed_dims),   # 256+256+1=513 → 256
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.Sigmoid(),                                  # 输出 gate ∈ [0,1]
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self,
                query_feat: Tensor,       # [B, Q, C=256], 当前query特征
                memory_output: Tensor,    # [B, Q, C=256], memory attention 输出 h_i
                query_confidence: Tensor, # [B, Q, 1], 当前query最大cls概率
                ) -> Tensor:              # [B, Q, C=256], 融合后的query特征
        # Gate
        gate_input = torch.cat([query_feat, memory_output, query_confidence], dim=-1)
                                                          # [B, Q, 513]
        gate = self.gate_mlp(gate_input)                  # [B, Q, 256], values in [0,1]

        # Gated residual
        enhanced = query_feat + gate * self.ffn(memory_output)  # [B, Q, 256]
        return self.norm(enhanced)                              # [B, Q, 256]
```

### F.5 完整 Memory Read 流程伪代码

```python
def memory_read(self, curr_query_feat, curr_query_pos, curr_query_cls,
                img_metas):
    """在 forward_backbone 中调用, RAP 输出之后、SCF 循环之前。"""

    if not self.memory_enabled or len(self.query_memory_bank) == 0:
        return curr_query_feat  # 空记忆或关闭时直接返回

    # 1. 从 memory bank 读取所有历史条目
    mem_data = self.query_memory_bank.read_all()
    # mem_data['mem_feat']:      [B, K*Q_valid, 256]
    # mem_data['mem_pos']:       [B, K*Q_valid, 48, 3]
    # mem_data['mem_ego2global']: [K, B, 4, 4]
    # mem_data['mem_timestamps']: [K]

    # 2. Ego pose 对齐: 历史 query 位置 → 当前 ego 坐标系
    curr_ego2global = torch.tensor(img_metas[0]['ego2global']).cuda()
    aligned_mem_pos = self.ego_pose_aligner(
        mem_data['mem_pos'],
        mem_data['mem_ego2global'],  # 需要逐条目处理
        curr_ego2global,
    )  # [B, K*Q_valid, 48, 3], 当前ego系归一化坐标

    # 3. Memory Attention: 软检索
    h = self.causal_memory_attn(
        query_feat=curr_query_feat,          # [B, 720, 256]
        query_pos=curr_query_pos,            # [B, 720, 48, 3]
        mem_feat=mem_data['mem_feat'],       # [B, M_total, 256]
        mem_pos_aligned=aligned_mem_pos,     # [B, M_total, 48, 3]
        mem_confidence=mem_data['mem_confidence'],
        mem_time_delta=mem_data['mem_time_delta'],
        mem_valid_mask=mem_data['mem_valid_mask'],
    )  # [B, 720, 256]

    # 4. 门控残差融合
    curr_confidence = curr_query_cls.sigmoid().max(-1)[0].mean(-1, keepdim=True)
                                              # [B, 720, 1]
    enhanced_query_feat = self.gated_fusion(
        query_feat=curr_query_feat,           # [B, 720, 256]
        memory_output=h,                      # [B, 720, 256]
        query_confidence=curr_confidence,     # [B, 720, 1]
    )  # [B, 720, 256]

    return enhanced_query_feat
```

---

## G. 分阶段实现路线

### V0: 原始 SparseWorld 基线复现 ✅ 已完成
- L2: 0.160 / 0.195 / 0.239
- Box Col: 0.00095 / 0.00101 / 0.00111
- 结果已保存在 `Initial_Output.md`

### V1: 单次前向内部的观测记忆读取
**目标**: 验证 memory read 的管线正确性，尚不做 pose 对齐
**实现内容**:
1. 创建 `query_memory.py`，实现 `QueryMemoryBank` 和简化版 `CausalQueryMemoryAttention`（仅语义相似度，无距离/时间/置信度修正）
2. 修改 `sparseworld_4d_traj.py`：
   - `__init__` 新增 memory 模块（`memory_enabled` 控制）
   - `forward_backbone` 在 RAP→SCF 之间加入 memory read
   - 前向结束后调用 `_update_memory` 写入当前帧 query
3. 修改 `opus.py`: `simple_test_online` 中 scene 切换时清空 memory
4. **不做 pose 对齐**: 暂时假设相邻帧 ego 位移很小，先验证管线
5. **融合方式**: 简单加法 `curr_query_feat += alpha * h`，alpha=0.1

**验证**: shape test, 空 memory test, `memory_enabled=False` 一致性 test

### V2: 加入显式 ego-pose 对齐
**目标**: 正确实现坐标系变换
**实现内容**:
1. 实现 `EgoPoseAligner`
2. 在 memory write 时保存 `ego2global` 矩阵
3. 在 memory read 时对齐历史 query 位置到当前 ego 坐标系
4. 验证变换正确性：identity pose test, 已知平移/旋转 test

**验证**: identity pose 对齐 test, 单向平移/旋转 test

### V3: 加入置信度和时间衰减
**目标**: 实现完整的注意力分数公式
**实现内容**:
1. 完善 `CausalQueryMemoryAttention`：加入 spatial penalty, time decay, confidence bonus
2. 在 memory write 时计算并保存 confidence（cls_score sigmoid max）
3. 可选：spatial radius mask
4. 调参：λ_p, λ_t, λ_c

**验证**: 确认远距离 memory 被正确抑制, 低置信度 memory 权重降低

### V4: 加入门控残差融合
**目标**: 替换 V1 的简单加法为 learned gate
**实现内容**:
1. 实现 `ConfidenceGatedFusion`
2. 替换 V1 中的 `curr_query_feat += alpha * h` 为完整的门控融合
3. gate 初始化偏置为负值（初始时 gate ≈ 0，保持原始行为）

**验证**: gate 初始值 ≈ 0 test, 训练过程中 gate 统计, attention/gate 可视化

### V5: SCF 递归中加入临时预测记忆
**目标**: 在 SCF 每步预测前读取观测 + 预测 memory（方案 2）
**实现内容**:
1. 在 SCF 循环中，每步将当前 query 写入临时 prediction memory
2. Prediction memory 权重低于 observation memory（confidence *= 0.5）
3. Prediction memory 保留长度 ≤ 当前 interval（最多 6 步）
4. 必须 detach prediction memory 的 query_feat 和 query_pos
5. 训练时确保不跨 SCF 步传播梯度（通过 detach）
6. 推理时 batch_size=1，无跨 scene 问题

**验证**: 显存监控, 训练速度对比, SCF 内部各步 gate 统计

### V6: 完整训练、评估与可视化
**目标**: 端到端训练验证
**实现内容**:
1. 完整训练 64 epochs（复用原始 schedule）
2. 前 5 epochs pretrain 模式：memory 模块参与前向但不写入 memory（pretrain 阶段无时序）
3. Epoch 5+ finetune 模式：开始写入和读取 memory
4. 评估：mIoU, L2 error, collision rate
5. 可视化：attention heatmap, gate 分布, effective memory count

**注意**: 训练时每个 batch 是独立场景，memory 在 batch 开始时为空。memory 的价值主要体现在推理（在线推理跨帧累积 memory）。训练时需要在同一 scene 内的连续样本间传递 memory —— 这可能需要修改 dataloader 使同 scene 样本连续出现。

### 关于分阶段顺序的调整说明

原始建议的顺序（V1→V6）在代码分析后基本合理，无需调整。原因：

1. V1 验证管线完整性是必需的第一步
2. V2 的 pose 对齐在逻辑上独立于 V3/V4，适合单独验证
3. V3 的置信度/时间衰减是注意力公式的补全，不依赖 V4 的门控
4. V4 的门控融合是效果的关键增强，放在 V3 之后正确
5. V5 的预测 memory 复杂度最高，放在最后正确

**唯一需要注意的**: 训练阶段的 memory 写入/读取需要特殊处理（见 V6 注意事项）。SparseWorld 训练时每个样本独立，不同于 OccWorld 一次输入 16 帧。建议在 V1-V4 阶段仅在推理时启用 memory（训练时 memory 为空），V6 再解决训练时的 memory 传递问题。

---

## H. 验证方案

### H.1 Shape / Unit Test

```python
def test_query_memory_bank_shapes():
    bank = QueryMemoryBank(bank_size=5, embed_dims=256)
    B, Q, C, P = 2, 720, 256, 48
    for t in range(7):  # 写入7帧，测试环形淘汰
        bank.write(
            query_feat=torch.randn(B, Q, C),
            query_pos=torch.rand(B, Q, P, 3),
            cls_scores=torch.randn(B, Q, P, 17),
            ego2global=torch.eye(4).unsqueeze(0).expand(B, -1, -1),
            timestamp=t * 0.5,
        )
    assert len(bank) == 5  # 最旧的2帧被淘汰
    data = bank.read_all()
    assert data['mem_feat'].shape[0] == B
    assert data['mem_feat'].shape[2] == C
    # M_total = sum of valid queries across 5 frames
```

### H.2 Identity Pose 对齐测试

```python
def test_identity_pose_alignment():
    aligner = EgoPoseAligner(pc_range=[-40,-40,-1,40,40,5.4])
    pos = torch.rand(1, 100, 48, 3)  # 归一化坐标
    identity = torch.eye(4).unsqueeze(0)
    aligned = aligner(pos, identity, identity)
    assert torch.allclose(pos, aligned, atol=1e-5), "Identity pose should not change positions"
```

### H.3 平移和旋转变换测试

```python
def test_translation_alignment():
    aligner = EgoPoseAligner(pc_range=[-40,-40,-1,40,40,5.4])
    pos = torch.tensor([[[[0.5, 0.5, 0.5]]]])  # 中心点 (0, 0, 2.2) in ego
    # 历史帧ego在全局坐标 (0,0,0), 当前帧ego在全局坐标 (10,0,0)
    hist_ego2global = torch.eye(4).unsqueeze(0)
    curr_ego2global = torch.eye(4).unsqueeze(0)
    curr_ego2global[0, 0, 3] = 10.0  # x方向平移10米
    aligned = aligner(pos, hist_ego2global, curr_ego2global)
    # 历史点 (0,0,2.2) 在当前ego系中应为 (-10, 0, 2.2)
    decoded = decode_points(aligned, [-40,-40,-1,40,40,5.4])
    assert abs(decoded[0,0,0,0].item() - (-10.0)) < 0.01

def test_rotation_alignment():
    # 类似，构造90度旋转矩阵测试
    ...
```

### H.4 空记忆测试

```python
def test_empty_memory_passthrough():
    model = SparseWorld4DTraj(memory_enabled=True, ...)
    # 不写入任何 memory
    result = model.memory_read(query_feat, query_pos, query_cls, img_metas)
    assert torch.equal(result, query_feat), "Empty memory should return original features"
```

### H.5 Scene 切换与 Batch 隔离测试

```python
def test_scene_switch_clears_memory():
    model = SparseWorld4DTraj(memory_enabled=True, ...)
    # 模拟3帧推理
    for i in range(3):
        model.forward(...)
    assert len(model.query_memory_bank) == 3
    # 模拟 scene 切换
    model._clear_memory()
    assert len(model.query_memory_bank) == 0

def test_batch_isolation():
    # batch_size=1 during inference, 确认不同sample不共享memory
    # 这在 simple_test_online 中由 assert len(img_metas)==1 保证
    ...
```

### H.6 `memory_enabled=False` 一致性测试

```python
def test_baseline_consistency():
    model_orig = SparseWorld4DTraj(memory_enabled=False, ...)
    model_mem = SparseWorld4DTraj(memory_enabled=True, ...)
    # 加载相同权重
    model_mem.load_state_dict(model_orig.state_dict(), strict=False)
    # 空 memory 下前向
    img = torch.randn(1, 5, 6, 3, 256, 704)
    with torch.no_grad():
        out_orig = model_orig(img, img_metas)
        out_mem = model_mem(img, img_metas)
    assert torch.allclose(out_orig['semantic_occ_0s'], out_mem['semantic_occ_0s']),
        "With empty memory, outputs should match original baseline"
```

### H.7 前向/反向 Smoke Test

```python
def test_forward_backward_smoke():
    model = SparseWorld4DTraj(memory_enabled=True, ...)
    model.train()
    # 构造mini数据
    img = torch.randn(1, 5, 6, 3, 256, 704, requires_grad=True)
    losses = model.forward_train(img, img_metas, voxel_semantics, ...)
    total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
    total_loss.backward()
    # 确认所有新增参数有梯度
    for name, param in model.named_parameters():
        if 'memory' in name or 'gate' in name:
            assert param.grad is not None, f"No gradient for {name}"
```

### H.8 显存、速度与参数量统计

```python
# 显存
torch.cuda.reset_peak_memory_stats()
model(img, img_metas)
peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"Peak GPU memory: {peak_mem:.2f} GB")

# 速度
import time
times = []
for _ in range(10):
    start = time.time()
    model(img, img_metas)
    torch.cuda.synchronize()
    times.append(time.time() - start)
print(f"Inference time: {sum(times[2:])/len(times[2:]):.3f}s")  # skip warmup

# 参数量
total_params = sum(p.numel() for p in model.parameters())
memory_params = sum(p.numel() for n, p in model.named_parameters()
                    if 'memory' in n or 'gate' in n or 'aligner' in n)
print(f"Total params: {total_params/1e6:.1f}M")
print(f"Memory module params: {memory_params/1e6:.1f}M")
# 预估: W_q + W_k + W_v + out_proj = 4 × 256×256 = 262K
#        gate_mlp ≈ 513×256 + 256×256 ≈ 197K
#        ffn ≈ 256×512 + 512×256 ≈ 262K
#        总计 ≈ 0.72M 新增参数
```

### H.9 日志方案

在 `forward_backbone` 中加入如下 logging（训练每 100 iter, 推理每帧）:

```python
if self.memory_enabled and self.training and iter_count % 100 == 0:
    logger.info(f"[Memory] bank_size={len(self.query_memory_bank)}, "
                f"effective_entries={mem_valid_mask.sum().item()}, "
                f"gate_mean={gate.mean().item():.4f}, "
                f"gate_std={gate.std().item():.4f}, "
                f"attn_entropy={(-attn_weights * attn_weights.log()).sum(-1).mean().item():.4f}, "
                f"avg_confidence={mem_confidence.mean().item():.4f}, "
                f"max_time_delta={mem_time_delta.max().item():.2f}s")
```

---

## 附录: 现有 SparseWorld 中是否已存在相关机制

经代码验证，以下机制在现有 SparseWorld 中**不存在**:

| 机制 | 现有状态 | 说明 |
|------|---------|------|
| 历史 Query 缓存 | ❌ 不存在 | `query_feat` 每次从零向量开始 |
| 跨帧 Query 位姿对齐 | ❌ 不存在 | 只在 loss 计算时对 GT voxels 做对齐 |
| 置信度门控 | ❌ 不存在 | 无 query 级置信度筛选或加权 |
| 历史 Query 软检索 | ❌ 不存在 | ind_mask 仅约束单次前向内的时间步可见性 |
| Memory bank | ❌ 不存在 | `self.memory` 仅缓存图像 backbone 特征 |

**结论**: 本 Idea 的所有核心组件均为全新增量，不与现有代码功能重叠。
