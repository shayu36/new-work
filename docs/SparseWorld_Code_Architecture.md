# SparseWorld 代码架构全解析

> 基于 AAAI 2026 论文 "SparseWorld: 4D Occupancy World Model with Sparse Dynamic Queries"
> 本文档梳理项目中每个文件的功能，为后续迭代开发提供参考。

---

## 一、项目总览

```
SparseWorld/
├── mmdet3d/                    # 核心代码库
│   ├── models/                 # 模型定义
│   │   ├── sparsedetectors/    # ★ SparseWorld/OPUS 核心（新增）
│   │   ├── detectors/          # PreWorld 检测器系列（基线）
│   │   ├── heads/              # 占用预测头
│   │   ├── necks/              # 特征融合 neck（含视角变换）
│   │   ├── backbones/          # 图像骨干网络
│   │   ├── nerf/               # NeRF 渲染头
│   │   ├── modules/            # 可变形注意力模块
│   │   └── loss_utils/         # 损失函数
│   ├── datasets/               # 数据集与数据管线
│   ├── core/                   # 基础组件（bbox, eval, hook等）
│   ├── ops/                    # 算子（BEV pooling, PointNet等）
│   ├── apis/                   # 训练/测试 API
│   └── utils/                  # 工具函数
├── configs/                    # 训练/测试配置文件
│   ├── sparseworld/nuscenes-temporal/  # ★ SparseWorld 时序配置
│   ├── sparseworld/nuscenes/           # PreWorld 非时序配置
│   └── _base_/                         # 基础配置模板
├── tools/                      # 训练、测试、数据处理脚本
├── AD-MLP/                     # 轨迹规划评估模块
├── data/nuscenes/              # nuScenes 数据集（符号链接）
├── ckpts/                      # 模型权重
├── admlp/                      # AD-MLP 评估所需的 pkl 文件
├── occworld/                   # OccWorld 评估所需的 pkl 文件
└── docs/                       # 文档
```

---

## 二、模型继承体系

SparseWorld 建立在 BEVDet 系列检测器之上，继承链如下：

```
BaseDetector (mmdet)
  └── MVXTwoStageDetector        # 多模态两阶段检测器基类
       └── CenterPoint            # 点云检测器
            └── BEVDet             # BEV 检测器（单帧）
                 └── BEVDet4D      # 时序 BEV（多帧对齐）
                      └── BEVDepth4D     # 深度监督 BEV
                           └── BEVStereo4D    # 立体匹配 BEV
                                └── BEVStereo4DOCC  # + 3D占用预测
                                     └── PreWorld        # + NeRF预训练 [基线方法]
                                          └── PreWorld4DTraj  # + 时序轨迹规划 [基线方法]

MVXTwoStageDetector (独立分支)
  └── OPUS                       # OPUS 稀疏查询检测器
       └── SparseWorld4DTraj      # ★ SparseWorld 完整模型
```

**两条路线对比：**
| 特性 | PreWorld 路线 | SparseWorld 路线 |
|------|-------------|-----------------|
| 特征表示 | 稠密 3D 体素 | 稀疏 query 点 |
| 占用预测头 | OccHead (3D 卷积) | OPUSHead (Transformer) |
| 时序建模 | 递归体素特征更新 | 自回归 query 扩展 |
| 图像到3D | LSS view transform | 3D→2D 采样 |
| 计算效率 | 较低（稠密计算） | 较高（稀疏计算） |

---

## 三、核心模型文件详解

### 3.1 SparseWorld/OPUS 核心 (`mmdet3d/models/sparsedetectors/`)

这是 SparseWorld 论文的核心贡献代码。

#### `sparseworld_4d_traj.py` — 顶层模型
- **类**: `SparseWorld4DTraj`（继承 `OPUS`）
- **功能**: SparseWorld 完整模型，在 OPUS 占用预测基础上增加未来时空预测和自车轨迹规划
- **核心流程** (`forward_backbone`):
  1. `plan_head` 将自车运动状态编码为 ego 特征
  2. `points_scale_branch` 动态调整 query 点初始化尺度
  3. 调用 OPUS backbone 提取图像特征，通过 `pts_bbox_head` 得到当前帧 query 特征
  4. 按 `ind_stamps_all` 分离当前帧/未来各帧 query
  5. **自回归未来预测循环**（循环 `num_fu_frames` 次）：
     - ego 特征与场景 query 做交叉注意力 → 轨迹预测
     - 逐帧累积未来 query，通过 reg/vel/cls branch 预测位移和语义
     - 运动物体（类别 2-10）叠加速度偏移
- **关键子模块**: `plan_head`, `ego_cross_attn`, `traj_head`, `reg_branch`, `vel_branch`, `cls_branch`
- **迭代要点**: 修改未来预测策略、轨迹规划头、损失函数权重

#### `opus.py` — OPUS 基础检测器
- **类**: `OPUS`（继承 `MVXTwoStageDetector`）
- **功能**: 图像特征提取、数据增强、在线/离线推理逻辑
- **核心方法**:
  - `extract_img_feat`: 通过 backbone + neck 提取特征，支持 GridMask
  - `extract_feat`: 完整特征提取，含颜色增强、归一化、padding
  - `simple_test_online`: 在线推理（逐帧缓存特征，节省显存）
  - `simple_test_offline`: 离线推理（一次性处理所有帧）
- **迭代要点**: 修改数据增强策略、特征提取管线

#### `opus_head.py` — OPUS 检测头
- **类**: `OPUSHead`（`pts_bbox_head`）
- **功能**: 管理 query 点初始化、调用 Transformer、计算损失、生成占用预测
- **核心方法**:
  - `forward`: 初始化 query → Transformer 精炼 → 分配时间戳
  - `_get_target_single`: KNN 双向最近邻匹配（Chamfer Distance 思路）
  - `get_occ`: **推理核心** — 稀疏 query → 稠密体素，用 `scatter_max` 聚合，膨胀+腐蚀填空洞
  - `loss_stack` / `loss_future`: 当前帧/未来帧损失
  - `get_sparse_voxels`: 稠密 GT → 稀疏点云表示
  - `reset_mask`: 因果注意力 mask，防止未来 query 看到过去
- **关键参数**: `num_query`=720, `num_fu_query`=[720]*N, `empty_dist_thr`, `empty_weights`
- **迭代要点**: 修改 query 数量、匹配策略、稀疏→稠密聚合方式

#### `opus_transformer.py` — Transformer 架构
- **类**:
  - `OPUSTransformer`: 顶层封装
  - `OPUSTransformerDecoder`: 管理多层 DecoderLayer
  - `OPUSTransformerDecoderLayer`: 单层解码器流程
  - `OPUSSelfAttention`: **Scale-Adaptive Self Attention (SASA)** — 论文核心创新
  - `OPUSCrossAttention`: 距离自适应交叉注意力
  - `OPUSSampling`: 自适应时空采样
  - `AdaptiveMixing`: 自适应通道/点混合（借鉴 AdaMixer）
- **单层解码流程**: `位置编码 → 采样 → 混合 → 自注意力 → FFN → 分类+回归 → 精炼点坐标`
- **SASA 核心**: 计算 query 间欧氏距离，用学习的 τ 参数调制距离作为注意力偏置
- **迭代要点**: 修改注意力机制、采样策略、解码器层数

#### `opus_sampling.py` — 4D 时空采样
- **函数**: `sampling_4d`
- **功能**: 3D 采样点 → 投影到各相机视角 → 从多尺度 FPN 特征图双线性插值
- **关键优化**: 每个采样点只保留一个最佳视角（`argmax(valid_mask)`）
- **迭代要点**: 修改采样点数量、多视角融合策略

#### `utils.py` — 工具函数集
- `sparse2dense`: 稀疏体素 → 稠密张量
- `GridMask`: 网格掩码数据增强
- `GpuPhotoMetricDistortion`: GPU 光度畸变增强
- `pad_multiple`: 图像 padding 到整数倍
- `rotation_3d_in_axis`: 绕 z 轴旋转
- `inverse_sigmoid`: sigmoid 反函数

#### `checkpoint.py` — 梯度检查点
- PyTorch `torch.utils.checkpoint` 的自定义副本
- 被 SelfAttention/CrossAttention/Sampling/Mixing 广泛使用，以计算换显存

#### `bbox/` 子目录
| 文件 | 功能 |
|------|------|
| `bbox/utils.py` | `encode/decode_points`（归一化坐标转换）、`get_matched_inds`（query→时间步分配）、`trans_coords`（帧间变换）、`dist_loss_weight`（远距离衰减）、`MultiheadAttention` |
| `bbox/sdnetutils.py` | utils.py 的精简版本 |
| `bbox/assigners/hungarian_assigner_3d.py` | 匈牙利匹配器（3D 目标检测用，OPUS 占用预测不常用） |
| `bbox/coders/nms_free_coder.py` | 无 NMS BBox 解码器 |
| `bbox/match_costs/match_cost.py` | 3D BBox L1/BEV L1/IoU3D 匹配代价 |

#### `csrc/` 子目录
- `wrapper.py`: CUDA 多尺度多视角采样加速（`msmv_sampling`）
- `setup.py`: CUDA 扩展编译脚本
- 如果 CUDA 编译失败会降级到 PyTorch `F.grid_sample`

---

### 3.2 PreWorld 检测器系列 (`mmdet3d/models/detectors/`)

PreWorld 是基线方法，SparseWorld 在此基础上改进。

| 文件 | 类 | 功能 |
|------|-----|------|
| `base.py` | `Base3DDetector` | 3D 检测器基类 |
| `mvx_two_stage.py` | `MVXTwoStageDetector` | 多模态两阶段检测基类，管理 backbone/neck/head |
| `centerpoint.py` | `CenterPoint` | 点云检测器 |
| `bevdet.py` | `BEVDet`, `BEVDet4D`, `BEVDepth4D`, `BEVStereo4D` | BEV 检测器系列，实现 2D→3D 视角变换、时序对齐 |
| `bevdet_occ.py` | `BEVStereo4DOCC` | 在 BEVStereo4D 基础上加 3D 占用预测 |
| `preworld.py` | `PreWorld` | NeRF 渲染预训练 + OccHead 占用预测（两阶段训练） |
| `preworld_temporal_traj.py` | `PreWorld4DTraj` | PreWorld + 时序递归预测 + 轨迹规划 |
| `loss.py` | — | 占用预测损失函数：`CE_ssc_loss`, `sem_scal_loss`, `geo_scal_loss`, `l1_loss`, `l2_loss` |
| `lovasz_softmax.py` | — | Lovász-Softmax 损失（用于占用语义分割） |

#### `preworld.py` — PreWorld 检测器
- **两阶段训练**:
  - Stage 1 (`is_pretrain=True`): NeRF 体积渲染监督（2D 深度/语义/颜色）
  - Stage 2: OccHead 3D 占用预测（3D 语义分割）
- **核心方法**: `bev_encoder` 将 BEV 体素特征编码，`nerf_head` 做 NeRF 渲染，`occupancy_head` 做占用预测

#### `preworld_temporal_traj.py` — PreWorld4DTraj
- **时序递归**: 循环 `num_fu_frames` 帧，每帧用 `single_step_future_prediction` 更新体素特征
- **未来预测**: 体素特征 → ego state 融合 → 下一帧特征 → 占用预测
- **轨迹规划**: ego 特征 → `ego_query_attn`(与 BEV 特征交叉注意力) → `traj_branch` 输出 2D 轨迹

---

### 3.3 特征提取 (`mmdet3d/models/necks/`, `backbones/`)

| 文件 | 类 | 功能 |
|------|-----|------|
| `necks/view_transformer.py` | `LSSViewTransformer`, `LSSViewTransformerBEVDepth`, `LSSViewTransformerBEVStereo` | **LSS 视角变换**：2D 图像 → 3D 体素。深度估计 + 外积 + voxel pooling |
| `necks/fpn.py` | `CustomFPN` | 自定义 FPN，含 ASPP 模块 |
| `necks/lss_fpn.py` | `FPN_LSS` | LSS 专用 FPN |
| `necks/second_fpn.py` | `SECONDFPN` | SECOND 检测器的 FPN |
| `backbones/resnet.py` | `CustomResNet` | 自定义 ResNet（支持去除 stem stride） |
| `backbones/swin.py` | `SwinTransformer` | Swin Transformer backbone |

#### LSS 视角变换核心流程
```
多视角图像 [B,N,C,H,W]
  → 2D backbone 提取特征
  → 深度估计网络 → D 个深度 bin 的概率分布 [B,N,D,H,W]
  → 特征 × 深度概率 → 伪点云 [B,N,D,H,W,C]
  → 根据相机内外参投影到 3D 体素空间
  → BEV pooling 聚合 → 3D 体素特征 [B,C,X,Y,Z]
```

---

### 3.4 占用预测头 (`mmdet3d/models/heads/`)

| 文件 | 类 | 功能 |
|------|-----|------|
| `occupancy_head.py` | `OccHead` | 多尺度 3D 卷积占用预测头（PreWorld 用） |
| | `DownScaleModule3DCustom` | 3D 体素下采样模块 |

- **OccHead**: 4 层 3D 卷积，逐层下采样 → 上采样 → 残差连接，最终输出 [B, num_classes, X, Y, Z]

---

### 3.5 NeRF 渲染 (`mmdet3d/models/nerf/`)

| 文件 | 类 | 功能 |
|------|-----|------|
| `nerf_head.py` | `NerfHead` | NeRF 体积渲染头：从 3D 体素特征生成 2D 深度/语义/颜色监督信号 |
| `utils.py` | — | 射线生成 (`create_rays`)、体积渲染 (`volume_rendering`)、采样策略 |

- **NeRF 预训练流程**: 3D体素特征 → 沿射线采样 → MLP 预测(密度σ, 颜色c, 语义s) → 体积渲染 → 与 GT 2D图像/深度对比

---

### 3.6 损失函数与模块 (`mmdet3d/models/loss_utils/`, `modules/`)

| 文件 | 功能 |
|------|------|
| `loss_utils/focal_loss.py` | `CustomFocalLoss`（用于 OPUS 分类损失） |
| `modules/deformable_self_attn.py` | `DeformableSelfAttention`（可变形自注意力） |
| `modules/multi_scale_deformable_attn_function.py` | 多尺度可变形注意力的 CUDA 实现 |

---

## 四、数据集与数据管线

### 4.1 数据集类 (`mmdet3d/datasets/`)

| 文件 | 类 | 功能 |
|------|-----|------|
| `custom_3d.py` | `Custom3DDataset` | 3D 数据集基类 |
| `nuscenes_dataset.py` | `NuScenesDataset` | nuScenes 基础数据集 |
| `nuscenes_dataset_occ.py` | `NuScenesDatasetOccupancy` | + 占用 GT 加载（Occ3D gts） |
| `nuscenes_dataset_occ_trajectory.py` | `NuScenesDatasetOccupancyTrajectory` | ★ + 时序占用 + 轨迹 GT（加载未来 6 帧） |
| `occ_metrics.py` | `Metric_mIoU`, `Metric_mIoU_Temporal` | 占用预测评估指标（mIoU/IoU） |
| `ray.py` | `RayGenerator` | NeRF 射线生成器 |
| `builder.py` | — | 数据集构建器 |

#### `nuscenes_dataset_occ_trajectory.py` 核心
- 加载当前帧 + 未来 `num_fu_frames` 帧的占用 GT
- 计算帧间 ego motion 变换矩阵（用于坐标系对齐）
- 提供 GT 轨迹（自车未来 3s 内 6 个路径点）
- 被 SparseWorld 和 PreWorld4DTraj 共用

### 4.2 数据管线 (`mmdet3d/datasets/pipelines/`)

| 文件 | 功能 |
|------|------|
| `loading.py` | `PrepareImageInputs`（图像加载+增强）、`LoadOccupancy`（加载占用GT）、`CreateRays`（生成NeRF射线） |
| `loading_traj_temporal.py` | ★ `LoadOccupancyTemporalTrajectory`：加载时序占用+轨迹GT，含帧间变换 |
| `transforms_3d.py` | 3D 数据增强（翻转、旋转、缩放等） |
| `formating.py` | `DefaultFormatBundle3D`：数据格式化为 mmdet 标准格式 |
| `compose.py` | 管线组合器 |

---

## 五、配置文件详解

### 5.1 SparseWorld 时序配置 (`configs/sparseworld/nuscenes-temporal/`)

| 配置文件 | 用途 | 模型类 |
|----------|------|--------|
| `bevstereo-occ-traj.py` | BEVStereo 基线（时序占用+轨迹） | `PreWorld4DTraj` |
| `preworld-7frame-pretrain-traj.py` | PreWorld NeRF 预训练 | `PreWorld4DTraj` (is_pretrain=True) |
| `preworld-7frame-finetune-traj.py` | PreWorld 占用微调 | `PreWorld4DTraj` (is_pretrain=False) |
| **`sparseworld-traj-finetune.py`** | **★ SparseWorld 主配置** | `SparseWorld4DTraj` |
| `sparseworld-traj-finetune-72pts.py` | SparseWorld 72点变体 | `SparseWorld4DTraj` |

#### `sparseworld-traj-finetune.py` 关键参数

```python
model = dict(
    type='SparseWorld4DTraj',
    img_backbone=dict(type='ResNet', depth=50),       # ResNet-50 backbone
    pts_bbox_head=dict(
        type='OPUSHead',
        num_query=720,                                 # 每帧 query 数量
        num_fu_query=[720, 720, 720, 720, 720, 720],  # 未来各帧 query 数
        transformer=dict(
            type='OPUSTransformer',
            decoder=dict(
                num_layers=6,                          # 6 层 Transformer 解码器
                num_refines=[1,1,1,1,1,1],            # 每层精炼次数
            )
        ),
    ),
    num_fu_frames=6,                                   # 预测未来 6 帧
)
# 训练：64 epochs, AdamW lr=2e-4, CosineAnnealing
# 数据：时序7帧（1当前+6未来），ResNet-50 backbone
```

### 5.2 PreWorld 非时序配置 (`configs/sparseworld/nuscenes/`)

| 配置文件 | 用途 |
|----------|------|
| `bevstereo-occ.py` | BEVStereo 占用基线 |
| `preworld-7frame-pretrain.py` | PreWorld NeRF 预训练（非时序） |
| `preworld-7frame-finetune.py` | PreWorld 占用微调（非时序） |

### 5.3 基础配置 (`configs/_base_/`)

- `datasets/nus-3d.py`: nuScenes 数据集参数
- `default_runtime.py`: 日志、checkpoint 保存策略
- `schedules/cosine.py`: CosineAnnealing 学习率策略

---

## 六、训练与测试工具

### 6.1 核心脚本 (`tools/`)

| 文件 | 功能 |
|------|------|
| `train.py` | 单机训练入口 |
| `dist_train.sh` | 分布式训练启动脚本 |
| `test.py` | 单机测试入口（占用预测 mIoU + 生成 output_data.pkl） |
| `test_temporal.py` | 时序测试入口 |
| `dist_test.sh` / `dist_test_temporal.sh` | 分布式测试脚本 |
| `get_flops.py` | 计算模型 FLOPs |
| `create_data_bevdet.py` | 生成 BEVDet 格式数据 info pkl |

### 6.2 数据生成 (`tools/gen_data/`)

| 文件 | 功能 |
|------|------|
| `gen_depth_gt.py` | 生成深度 GT |
| `gen_seg_gt_from_lidarseg.py` | 从 lidarseg 生成语义分割 GT |
| `gen_seg_gt_from_occ.py` | 从 occupancy 生成语义分割 GT |

### 6.3 可视化 (`tools/visualization/`)

| 文件 | 功能 |
|------|------|
| `vis_tool.py` | 占用可视化工具 |
| `visual.py` | 通用可视化 |

---

## 七、评估模块 (`AD-MLP/`)

轨迹规划评估借用 STP3/UniAD 的评估框架。

| 文件 | 功能 |
|------|------|
| `AD-MLP/deps/stp3/evaluate_for_mlp.py` | 评估入口：加载预测轨迹，计算 L2/碰撞率 |
| `AD-MLP/deps/stp3/stp3/planning_metrics.py` | `PlanningMetric` 类：L2 误差、点碰撞、BBox 碰撞 |

**评估指标**:
- `plan_L2_{1,2,3}s`: 1/2/3 秒的轨迹 L2 误差
- `plan_obj_col_{1,2,3}s`: 点级物体碰撞率
- `plan_obj_box_col_{1,2,3}s`: BBox 级物体碰撞率

**评估流程**:
1. `tools/test.py` 推理生成 `admlp/output_data.pkl`（token → 预测轨迹）
2. `evaluate_for_mlp.py` 加载 `output_data.pkl` + GT，计算规划指标

---

## 八、数据依赖文件

| 目录/文件 | 内容 |
|-----------|------|
| `data/nuscenes/samples/` | 关键帧图像 (~53G) |
| `data/nuscenes/sweeps/` | 中间帧图像 (~342G) |
| `data/nuscenes/v1.0-trainval/` | 元数据 JSON |
| `data/nuscenes/maps/` | 地图数据 |
| `data/nuscenes/gts/` | Occ3D 占用 GT |
| `data/nuscenes/bevdetv2-nuscenes_infos_{train,val}.pkl` | BEVDet 格式数据 info |
| `occworld/nuscenes_infos_{train,val}_temporal_v3_scene.pkl` | 时序场景 info |
| `admlp/fengze_nuscenes_infos_{train,val}.pkl` | AD-MLP 格式数据 info |
| `admlp/stp3_val/` | 评估所需 GT（filter_token, stp3_occupancy, stp3_traj_gt） |
| `ckpts/epoch_56.pth` | SparseWorld 预训练权重 |

---

## 九、核心数据流

### 9.1 训练数据流

```
nuScenes 数据
  ↓ PrepareImageInputs (加载6视角图像 + 增强)
  ↓ LoadOccupancyTemporalTrajectory (加载当前帧+未来6帧占用GT + 轨迹GT)
  ↓ DefaultFormatBundle3D (格式化)
  ↓
SparseWorld4DTraj.forward_train()
  ↓ extract_feat() → 多尺度图像特征 [B, TN, C, H, W]
  ↓ pts_bbox_head.forward() → 初始化720个query点 → OPUSTransformer 6层精炼
  ↓                           ↓ 每层: 3D→2D采样 → AdaptiveMixing → SASA → FFN
  ↓                           ↓ 输出: query_feat, cls_scores, refine_pts
  ↓ forward_backbone() → 自回归未来预测循环(6帧)
  ↓                       ↓ ego特征 × 场景query → 轨迹预测
  ↓                       ↓ reg/vel/cls branch → 未来query位置+语义
  ↓ 计算损失:
  ↓   - 当前帧: cls_loss (Focal) + pts_loss (L1)
  ↓   - 未来帧: fu_cls_loss + fu_pts_loss (每帧)
  ↓   - 轨迹:  traj_loss (L2)
```

### 9.2 推理数据流

```
测试图像序列
  ↓ simple_test_online() → 逐帧提取特征并缓存
  ↓ pts_bbox_head.forward() → Transformer 精炼 query
  ↓ forward_backbone() → 自回归预测未来6帧
  ↓ get_occ() → 稀疏query → 稠密体素占用 [B, 200, 200, 16]
  ↓            (sigmoid激活 → 分数过滤 → scatter_max聚合 → 膨胀填洞)
  ↓ 输出: semantic_occ_{0-6}s, geo_occ_{0-6}s, pred_traj
  ↓
evaluate_for_mlp.py → 计算 L2/碰撞率评估指标
```

---

## 十、迭代开发指南

### 10.1 常见修改点

| 修改目标 | 涉及文件 | 说明 |
|----------|----------|------|
| 增加/减少 query 数量 | `opus_head.py`, 配置文件 | 修改 `num_query`, `num_fu_query` |
| 修改 Transformer 层数 | `opus_transformer.py`, 配置文件 | 修改 `num_layers`, `num_refines` |
| 修改注意力机制 | `opus_transformer.py` | `OPUSSelfAttention`, `OPUSCrossAttention` |
| 修改采样策略 | `opus_sampling.py` | `sampling_4d`, 采样点数/视角选择 |
| 修改损失函数 | `opus_head.py`, `sparseworld_4d_traj.py`, `loss.py` | 各种 loss 权重和类型 |
| 修改时序预测帧数 | `sparseworld_4d_traj.py`, 配置文件 | `num_fu_frames`, `num_fu_query` |
| 修改轨迹规划 | `sparseworld_4d_traj.py` | `plan_head`, `traj_head`, `ego_cross_attn` |
| 修改 backbone | 配置文件 | `img_backbone` 替换为 Swin 等 |
| 修改数据增强 | `opus.py`, `utils.py`, `pipelines/` | GridMask, PhotoMetric, 3D变换 |
| 添加新的评估指标 | `occ_metrics.py`, `evaluate_for_mlp.py` | 新增 Metric 类 |

### 10.2 训练命令

```bash
# SparseWorld 训练 (8 GPU)
bash ./tools/dist_train.sh ./configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py 8

# SparseWorld 测试 (占用指标)
python tools/test.py --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py --checkpoint ckpts/epoch_56.pth

# 轨迹规划评估
cd AD-MLP/deps/stp3 && python evaluate_for_mlp.py
```

### 10.3 两阶段训练流程

```
Stage 1: NeRF 预训练 (preworld-7frame-pretrain-traj.py)
  → 学习从2D图像提取3D表示的能力
  → 监督信号: 2D 深度、语义、颜色

Stage 2: 占用微调 (sparseworld-traj-finetune.py)
  → 加载 Stage 1 权重
  → 端到端训练占用预测 + 轨迹规划
  → 监督信号: 3D 占用GT、轨迹GT
```

---

## 十一、16 类语义标签 (nuScenes-Occ3D)

| ID | 类别 | 运动性 |
|----|------|--------|
| 0 | others | 静态 |
| 1 | barrier | 静态 |
| 2 | bicycle | **动态** |
| 3 | bus | **动态** |
| 4 | car | **动态** |
| 5 | construction_vehicle | **动态** |
| 6 | motorcycle | **动态** |
| 7 | pedestrian | **动态** |
| 8 | traffic_cone | 静态 |
| 9 | trailer | **动态** |
| 10 | truck | **动态** |
| 11 | driveable_surface | 静态 |
| 12 | other_flat | 静态 |
| 13 | sidewalk | 静态 |
| 14 | terrain | 静态 |
| 15 | manmade | 静态 |
| 16 | vegetation | 静态 |
| 17 | empty (free) | — |

> 在 `sparseworld_4d_traj.py` 中，类别 2-10 被视为运动物体，会叠加速度偏移进行未来预测。
