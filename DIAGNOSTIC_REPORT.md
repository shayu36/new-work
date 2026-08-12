# STAC-QM Diagnostic Report

## 任务背景

在 SparseWorld (4D occupancy world model, Occ3D-nuScenes) 上加入 STAC-QM
(Spatio-Temporal Attention Confidence Query Memory) 模块，用于跨帧记忆增强。
当前需要诊断:Memory 模块是否正常工作、与 Baseline 的对比是否同口径。

## 代码位置

| 文件 | 作用 |
|------|------|
| `mmdet3d/models/sparsedetectors/query_memory.py` | STAC-QM 核心 (QueryMemoryBank / CausalQueryMemoryAttention / ConfidenceGatedFusion / STACQueryMemory) |
| `mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py` | 主模型 SparseWorld4DTraj，Memory 集成点 |
| `mmdet3d/datasets/pipelines/loading_query_memory.py` | 训练/测试 pipeline 中加载缓存 (LoadQueryMemoryFromFiles) |
| `tools/query_memory/precompute_query_memory.py` | 用已训练 checkpoint 预计算每帧 query 缓存 |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py` | STAC-QM 配置 (source='cache', freeze_base_model=True) |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory.py` | 旧版 memory 配置 (source='online') |

## Baseline 指标（按 SparseWorld 论文 Table 1）

论文: "SparseWorld: A Flexible, Adaptive, and Efficient 4D Occupancy World Model
Powered by Sparse and Dynamic Queries" (arXiv:2510.17482)

| 指标 | 1s | 2s | 3s | Avg |
|------|----|----|----|-----|
| mIoU | 14.93 | 13.15 | 11.51 | 13.20 |
| IoU | 22.96 | 22.10 | 21.05 | 22.03 |

对比方法 (论文 Table 1):
- OccWorld-D: mIoU 11.55/8.66/6.98, IoU 18.90/16.26/14.43
- PreWorld: mIoU 11.69/8.72/6.77, IoU 23.01/20.79/18.84
- PreWorld + Pre-training: mIoU 12.27/9.24/7.15, IoU 23.62/21.76/19.63

论文没有报告 0s (当前帧) 指标，仅报告未来 1s/2s/3s。

## 本地复现 Baseline (无 Memory, epoch 51)

训练配置: `sparseworld-traj-memory.py` (memory 未启用，实际是无 memory 基线)
checkpoint: `work_dirs/stacqm/epoch_51.pth`

| 指标 | 0s | 1s | 2s | 3s |
|------|----|----|----|-----|
| mIoU | 14.72 | 12.32 | 11.10 | 9.84 |
| IoU | 24.28 | 21.93 | 21.27 | 20.35 |

17 类逐类别 IoU (epoch 51, 0s):
others 2.64, barrier 12.99, bicycle 1.59, bus 11.46, car 13.79,
construction_vehicle 5.36, motorcycle 1.26, pedestrian 1.87, traffic_cone 2.9,
trailer 6.02, truck 9.91, driveable_surface 27.24, other_flat 16.31,
sidewalk 15.4, terrain 14.54, manmade 9.53, vegetation 14.43

完整逐 epoch 记录: `work_dirs/stacqm/eval_results/summary.csv` 和
`work_dirs/stacqm/eval_results_w4_34_52/summary_34_52.csv`

## 缓存覆盖率统计

| 指标 | train | val |
|------|-------|-----|
| 缓存文件 | 19,730 / 19,730 (100%) | 4,219 / 4,219 (100%) |
| 每帧 Query 数 | 256 (满) | 256 (满) |
| Confidence min/median/max | 0.608 / 0.938 / 0.994 | 0.564 / 0.915 / 0.995 |
| Conf ≥ 0.35 比例 | 100% | 100% |

- 缓存由 `precompute_query_memory.py` 用 epoch_51 checkpoint 生成
- train 缓存目录: `data/query_memory/sparseworld_train` (未上传, 19,730 个 .pt, 共 ~8GB)
- val 缓存目录: `data/query_memory/sparseworld_val` (未上传, 4,219 个 .pt)
- 抽样样本: 本仓库 `cache_samples/train/` 和 `cache_samples/val/` 各 20 个 .pt

## 缓存样本格式 (schema_version=1)

每个 .pt 文件包含:
- `schema_version`: 1
- `sample_idx`, `scene_id`, `frame_idx`, `timestamp`
- `ego2global`: [4,4] float32
- `query_feat`: [M, 256] float32 (M ≤ 256)
- `query_points_metric`: [M, 48, 3] float32 (米制坐标, pc_range=[-40,-40,-1,40,40,5.4])
- `query_conf`: [M] float32
- `valid_mask`: [M] bool
- `pc_range`, `embed_dims`, `num_points`, `num_classes`
- `source_config`, `source_checkpoint`

## 训练日志

- 训练日志: `work_dirs/stacqm/20260810_074239.log` (未上传, 1.8MB)
- 续训起点: epoch_33.pth, iter 81411
- 训练命令: `tools/dist_train.sh` (torchrun, 2x RTX 3090)
- 学习率: 5e-5 (sparseworld-traj-memory.py 配置), batch 4/GPU, CosineAnnealing + linear warmup
- 当前训练到 epoch 52, 每 epoch 2467 iter
- 总 loss ~12.9, grad_norm ~11.5

## 已知问题

1. **训练时 Memory 完全跳过**: `sparseworld_4d_traj.py` 中 `_query_memory_context`
   在 `self.training` 时返回 None, STAC-QM 参数被冻结 (requires_grad=False)。
   训练阶段 Memory 从不参与 forward。

2. **strict=False 静默降级**: `loading_query_memory.py` 默认 strict=False,
   缓存缺失时静默产生全零 memory tensor, 不报错。此前 train/val/test 全部指向
   `sparseworld_train` 目录, val 评估实际读取 train 的缓存 (现已分开生成)。

3. **checkpoint 兼容**: epoch_51.pth 包含 query_memory.* 权重 (STAC-QM 结构存在
   但冻结), precompute 脚本加载时会出现 unexpected keys 警告 (无害)。

4. **freeze_base_model=True**: stacqm 配置冻结全部基座参数, 只允许训练
   query_memory.* 参数, 但当前没有任何训练脚本实际训练这些参数。

## 需要诊断的问题

1. STAC-QM 在 eval 时是否真正改变了预测? 需要同口径对比:
   - Baseline: `sparseworld-traj-finetune.py` (无 memory) + epoch_51
   - Memory: `sparseworld-traj-finetune-stacqm.py` (cache memory) + epoch_51
   两者应使用相同评估脚本 (`tools/test_temporal.py`)、相同数据划分、相同 checkpoint。

2. 缓存中的 query 来自 `all_refine_pts[-1]` (num_refines=48 的最后一层),
   而 eval 时当前帧 query 也来自同一层, 但经过 `_apply_query_memory_once`
   融合的位置是 forecast 循环内。确认对齐正确性。

3. 缓存时间戳单位: 预计算用 `float(info['timestamp'])/1e6` (微秒转秒),
   确认 eval 时 meta timestamp 单位一致。
