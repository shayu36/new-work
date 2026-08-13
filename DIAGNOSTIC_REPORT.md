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
评估脚本: `tools/test_temporal.py`, 4,219 个 val 样本

| 指标 | 0s | 1s | 2s | 3s |
|------|----|----|----|-----|
| mIoU | 14.72 | 12.32 | 11.10 | 9.84 |
| IoU | 24.28 | 21.93 | 21.27 | 20.35 |

17 类逐类别 IoU (epoch 51):

| 类别 | 0s | 1s | 2s | 3s |
|------|----|----|----|-----|
| others | 2.87 | 2.79 | 2.67 | 2.64 |
| barrier | 17.46 | 16.14 | 14.90 | 12.99 |
| bicycle | 4.56 | 2.90 | 2.35 | 1.59 |
| bus | 22.11 | 18.40 | 14.44 | 11.46 |
| car | 23.98 | 19.41 | 16.06 | 13.79 |
| construction_vehicle | 7.71 | 7.21 | 6.58 | 5.36 |
| motorcycle | 4.93 | 2.62 | 1.82 | 1.26 |
| pedestrian | 7.75 | 4.83 | 2.98 | 1.87 |
| traffic_cone | 6.01 | 4.91 | 3.95 | 2.90 |
| trailer | 8.41 | 7.58 | 6.69 | 6.02 |
| truck | 16.19 | 13.90 | 11.79 | 9.91 |
| driveable_surface | 36.78 | 30.01 | 29.03 | 27.24 |
| other_flat | 23.23 | 18.38 | 17.74 | 16.31 |
| sidewalk | 21.70 | 17.53 | 16.73 | 15.40 |
| terrain | 19.78 | 16.74 | 15.81 | 14.54 |
| manmade | 10.70 | 10.35 | 10.03 | 9.53 |
| vegetation | 16.12 | 15.73 | 15.17 | 14.43 |
| free | 91.82 | 91.19 | 91.17 | 91.16 |

完整逐 epoch 记录: `work_dirs/stacqm/eval_results/summary.csv` 和
`work_dirs/stacqm/eval_results_w4_34_52/summary_34_52.csv`

## 训练信息 (日志 20260810_074239.log)

- **训练命令**: `tools/dist_train.sh` → torchrun 2 GPU (RTX 3090), master_port 29500,
  launcher=pytorch, config=`sparseworld-traj-memory.py`, work_dir=`work_dirs/stacqm`
- **checkpoint**: 从 `epoch_33.pth` 续训 (resumed epoch 33, iter 81411)
- **missing/unexpected keys**: 续训无此输出 (完整 checkpoint 恢复)
- **参数统计** (epoch_51.pth):
  - 总参数: 76,946,522
  - query_memory.* 参数: 856,612 (25 个 tensor, 训练时 frozen, requires_grad=False)
  - 基座参数: 76,089,910
- **学习率**: 5e-5, CosineAnnealing + linear warmup (warmup_iters=200), min_lr_ratio=1e-3
- **batch size**: 4/GPU × 2 GPU = 8
- **每 epoch**: 2,467 iter, 已训练至 epoch 52
- **总 loss 趋势** (各 epoch 首 iter 50/2467):
  13.35 (ep34) → 13.20 (ep35) → 13.01 (ep36) → 13.13 (ep37) → 12.99 (ep38)
  → 12.92 (ep42) → 12.70 (ep47) → 12.91 (ep48) → 12.85 (ep49) → 12.97 (ep50)
  → 12.84 (ep51) → 12.92 (ep52)
- **loss 组成** (ep52 iter250): init_loss_pts 2.83, d0-d5 loss_cls 0.28→0.15,
  d0-d5 loss_pts 1.89→0.44, fu1-fu6 loss_cls 0.15→0.18, fu1-fu6 loss_pts 0.43→0.46,
  loss_traj_1s-6s 0.038/0.020/0.022/0.028/0.033/0.050, grad_norm 11.2
- 日志摘录: `logs/train_log_head.txt` (本仓库已提交)

## 缓存覆盖率统计

| 指标 | train | val |
|------|-------|-----|
| 缓存文件 | 19,730 / 19,730 (100%) | 4,219 / 4,219 (100%) |
| 每帧 Query 数 | 256 (满) | 256 (满) |
| Confidence min/median/max | 0.608 / 0.938 / 0.994 | 0.564 / 0.915 / 0.995 |
| Conf ≥ 0.35 比例 | 100% | 100% |
| 本帧缓存命中率 | 100% | 100% |
| 有效历史帧数 mean | 2.48 | 2.48 |
| 0 历史帧比例 | 14.9% | 15.0% |
| 3 历史帧比例 | 80.1% | 80.1% |

历史帧统计口径: 模拟 `loading_query_memory.py` 的查找逻辑 (同 scene、
timestamp 严格小于当前帧、最多向前 10 帧内找 3 帧)。

- 缓存由 `precompute_query_memory.py` 用 epoch_51 checkpoint 生成
- train 缓存目录: `data/query_memory/sparseworld_train` (未上传, 19,730 个 .pt, 共 ~8GB)
- val 缓存目录: `data/query_memory/sparseworld_val` (未上传, 4,219 个 .pt)
- 抽样样本: 本仓库 `cache_samples/train/` 和 `cache_samples/val/` 各 20 个 .pt

### confidence / age / distance 分布

- **confidence**: train min=0.608 median=0.938 max=0.994; val min=0.564 median=0.915 max=0.995
  (缓存写入阈值 0.35, top-256 截断, 分布集中在高置信区间)
- **age** (eval 实测, 23.3M query): mean=0.879s std=0.263 max=1.70s
  (max_age=3.0s 未截断, 全部为 0.5s 间隔的相邻帧)
- **distance** (eval 实测): mean=7.74m std=2.24 max=12.0m
  (spatial_radius=12.0m 边界处有截断)

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

## Memory 评估结果 (epoch_51, val 4,219 样本, 2026-08-12)

评估配置: `sparseworld-traj-finetune-stacqm-val.py` (val 缓存 +
log_diagnostics=True), 评估脚本 `tools/test_temporal.py --eval segm`

| 指标 | Baseline (无Memory) | Memory (cache) | 差异 |
|------|--------------------|----------------|------|
| mIoU@0s | 14.72 | 14.72 | 0 |
| mIoU@1s | 12.32 | 12.32 | 0 |
| mIoU@2s | 11.10 | 11.10 | 0 |
| mIoU@3s | 9.84 | 9.84 | 0 |
| IoU@0s | 24.28 | 24.28 | 0 |
| IoU@1s | 21.93 | 21.93 | 0 |
| IoU@2s | 21.27 | 21.27 | 0 |
| IoU@3s | 20.35 | 20.35 | 0 |

17 类逐类别 IoU: Memory 与 Baseline 完全相同 (逐类别也零差异)。

### Memory 内部诊断量 (memory_eval_ep51_memory_diag.json, 23.3M query 样本)

| 诊断量 | mean | std | min | max | 解读 |
|--------|------|-----|-----|-----|------|
| has_candidate | 0.941 | 0.237 | 0 | 1 | 94% query 有候选 ✅ |
| candidate_count | 110.7 | 90.8 | 0 | 628 | 空间半径内候选充足 ✅ |
| topk_candidate_count | 27.5 | 9.8 | 0 | 32 | topk=32 平均选中 27.5 ✅ |
| support_conf | 0.837 | 0.218 | 0 | 0.991 | 支持帧置信度正常 ✅ |
| avg_distance | 7.74m | 2.24 | 0 | 12.0 | 空间距离合理 ✅ |
| avg_age | 0.879s | 0.263 | 0 | 1.70 | 时间因果性正常 ✅ |
| **avg_gate** | **0.0175** | **0.0044** | **0** | **0.0194** | ❌ gate 恒接近 0 |

### 根因分析: Memory 输出恒等于 Baseline

`ConfidenceGatedFusion` 初始化:
- `out_proj.weight` / `out_proj.bias` 零初始化 (`nn.init.zeros_`)
- `gate_mlp` 末层 bias 初始化为 -4.0 → gate = sigmoid(-4) ≈ 0.018
- STAC-QM 全部参数训练时冻结 (`requires_grad=False`),
  且训练时 `_query_memory_context` 直接返回 None (memory 从不参与训练 forward)

因此:
```
fused = query_feat + has_candidate × gate × out_proj(memory_output)
      = query_feat + has_candidate × 0.018 × 0
      = query_feat   (精确恒等)
```

**Memory 数学上恒等于 Baseline。** 基础设施 (缓存生成/加载/对齐/注意力候选筛选)
全部正常 (诊断量 has_candidate=94%, candidate_count=110), 但 STAC-QM
从未被训练, 冻结在初始化状态, 输出被零初始化设计为恒等 fallback。

### Q/K/V 梯度范数

N/A — STAC-QM 参数 `requires_grad=False`, 梯度恒为 0。且训练时
`_query_memory_context` 返回 None, memory 分支不参与计算图。

### 修复方向

1. 解冻 `query_memory.*` 参数 (移除 `param.requires_grad = False`)
2. 训练时让 memory 参与 forward (移除 `if self.training: return None`)
3. 跑 STAC-QM 微调训练 (freeze_base_model=True 或全参数微调)
4. 重新评估对比 Baseline
