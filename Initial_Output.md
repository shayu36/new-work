# SparseWorld Inference & Evaluation Results

## Run Information

- Date: 2026-07-30
- Server: JXY-3090 (2x NVIDIA GeForce RTX 3090, 24GB each)
- Python Environment: /data/jxy/projects/env (Python 3.9, conda)
- Config: configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py
- Checkpoint: ckpts/epoch_56.pth
- Dataset: nuScenes v1.0-trainval (validation split)

## Step 1: Occupancy Prediction Inference

### Command

```bash
conda activate /data/jxy/projects/env
cd /data/jxy/projects
python tools/test.py --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py --checkpoint ckpts/epoch_56.pth
```

### Output

- Total validation samples: 4219
- Speed: ~3.7-3.8 task/s
- Total time: ~20 minutes
- Output file: /data/jxy/projects/admlp/output_data.pkl
- Output trajectory shape per sample: (1, 6, 2) - 6 future timesteps, 2D (x, y)

## Step 2: Planning Metrics Evaluation

### Command

```bash
cd /data/jxy/projects/AD-MLP/deps/stp3
python evaluate_for_mlp.py
```

### Raw Output

```
plan_obj_col_1s : 0.0
plan_obj_box_col_1s : 0.0009480919688940048
plan_L2_1s : 0.1595965325832367
plan_obj_col_2s : 0.0
plan_obj_box_col_2s : 0.0010073477169498801
plan_L2_2s : 0.1945464313030243
plan_obj_col_3s : 0.0
plan_obj_box_col_3s : 0.0011061072582378983
plan_L2_3s : 0.2393464893102646
```

### Results Summary

| Metric            | 1s      | 2s      | 3s      |
|-------------------|---------|---------|---------|
| L2 (m)            | 0.1596  | 0.1945  | 0.2393  |
| Obj Collision (%) | 0.0000  | 0.0000  | 0.0000  |
| Box Collision (%) | 0.0948  | 0.1007  | 0.1106  |

### Notes

- Evaluation used 4219 out of 4819 filter tokens (600 tokens skipped due to missing predictions)
- Evaluation script modified to skip missing tokens and pad 2D trajectories to 3D for metric compatibility
- Modified files:
  - AD-MLP/deps/stp3/evaluate_for_mlp.py: updated output_data.pkl path, added token skip logic, added 2D->3D padding
  - AD-MLP/deps/stp3/stp3/planning_metrics.py: fixed torchmetrics compatibility (compute_on_step removal, import path)

## Dataset Configuration

```
data/nuscenes/
  samples/    -> symlink to nuscenes_raw/OpenDataLab___nuScenes/raw/Trainval/samples (53G)
  sweeps/     -> symlink to nuscenes_raw/OpenDataLab___nuScenes/raw/Trainval/sweeps (342G)
  v1.0-trainval/ -> symlink to nuscenes_raw/OpenDataLab___nuScenes/raw/Trainval/v1.0-trainval (2.5G)
  maps/       -> symlink to nuscenes_raw/OpenDataLab___nuScenes/raw/test/maps (5.6M)
  gts/        -> symlink to data/occ3d_gts/gts
  bevdetv2-nuscenes_infos_train.pkl (350M)
  bevdetv2-nuscenes_infos_val.pkl (71M)
```

## Dependencies Installed During Run

- ipython
- thop
- fvcore
- timm
