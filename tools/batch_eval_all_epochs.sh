#!/bin/bash
# Batch evaluation of all epoch checkpoints (single GPU - required by STAC-QM)
# Usage: bash tools/batch_eval_all_epochs.sh [gpu_id]

set -e

CONFIG="configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory.py"
WORK_DIR="./work_dirs/stacqm"
RESULT_DIR="${WORK_DIR}/eval_results"
GPU_ID=${1:-0}

LOCK_FILE="/tmp/batch_eval_stacqm.lock"
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "ERROR: Another instance is already running (PID $LOCK_PID). Exiting."
        exit 1
    else
        echo "WARNING: Stale lock file found (PID $LOCK_PID no longer running). Removing."
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

mkdir -p "$RESULT_DIR"
SUMMARY_FILE="${RESULT_DIR}/summary.csv"

echo "epoch,mIoU_1s,IoU_1s,mIoU_2s,IoU_2s,mIoU_3s,IoU_3s,mIoU_0s,IoU_0s" > "$SUMMARY_FILE"

for ckpt in $(ls ${WORK_DIR}/epoch_*.pth | sort -V); do
    epoch=$(basename "$ckpt" .pth | sed 's/epoch_//')
    LOG_FILE="${RESULT_DIR}/epoch_${epoch}_eval.log"

    if [ -f "$LOG_FILE" ] && grep -q "IoU" "$LOG_FILE" 2>/dev/null; then
        echo "Epoch $epoch already evaluated, skipping."
        continue
    fi

    echo "============================================"
    echo "Evaluating Epoch $epoch: $ckpt (GPU $GPU_ID)"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    /data/jxy/projects/env/bin/python3.9 \
        tools/test_temporal.py \
        "$CONFIG" \
        "$ckpt" \
        --eval segm \
        --gpu-id 0 \
        2>&1 | tee "$LOG_FILE"

    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Extract IoU values
    IOU_0S=$(grep "IoU at 0s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    IOU_1S=$(grep "IoU at 1s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    IOU_2S=$(grep "IoU at 2s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    IOU_3S=$(grep "IoU at 3s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)

    # Extract mIoU values
    MIOU_0S=$(grep "mIoU.*at 0s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    MIOU_1S=$(grep "mIoU.*at 1s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    MIOU_2S=$(grep "mIoU.*at 2s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    MIOU_3S=$(grep "mIoU.*at 3s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)

    echo "=> mIoU@0s: $MIOU_0S, mIoU@1s: $MIOU_1S, mIoU@2s: $MIOU_2S, mIoU@3s: $MIOU_3S"
    echo "=> IoU@0s: $IOU_0S, IoU@1s: $IOU_1S, IoU@2s: $IOU_2S, IoU@3s: $IOU_3S"
    echo "$epoch,$MIOU_1S,$IOU_1S,$MIOU_2S,$IOU_2S,$MIOU_3S,$IOU_3S,$MIOU_0S,$IOU_0S" >> "$SUMMARY_FILE"

    echo ""
done

echo "============================================"
echo "Evaluation complete!"
echo "============================================"
echo ""
echo "Summary:"
column -t -s',' "$SUMMARY_FILE"
