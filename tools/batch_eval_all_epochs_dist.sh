#!/bin/bash
# Batch distributed evaluation of all epoch checkpoints using 2 GPUs
# Usage: bash tools/batch_eval_all_epochs_dist.sh

set -e

CONFIG="configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory.py"
WORK_DIR="./work_dirs/stacqm"
RESULT_DIR="${WORK_DIR}/eval_results"
GPUS=2
MASTER_PORT=29501

mkdir -p "$RESULT_DIR"
SUMMARY_FILE="${RESULT_DIR}/summary.csv"

echo "epoch,mIoU_1s,IoU_1s,mIoU_2s,IoU_2s,mIoU_3s,IoU_3s,mIoU_avg" > "$SUMMARY_FILE"

for ckpt in $(ls ${WORK_DIR}/epoch_*.pth | sort -V); do
    epoch=$(basename "$ckpt" .pth | sed 's/epoch_//')
    echo "============================================"
    echo "Evaluating Epoch $epoch: $ckpt (2 GPUs)"
    echo "============================================"

    LOG_FILE="${RESULT_DIR}/epoch_${epoch}_eval.log"

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    /data/jxy/projects/env/bin/torchrun \
        --nproc_per_node=$GPUS \
        --master_port=$MASTER_PORT \
        tools/test_temporal.py \
        "$CONFIG" \
        "$ckpt" \
        --eval segm \
        --launcher pytorch \
        2>&1 | tee "$LOG_FILE"

    # Extract IoU values for each future time step
    IOU_1S=$(grep "IoU at 1s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    IOU_2S=$(grep "IoU at 2s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    IOU_3S=$(grep "IoU at 3s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)

    # Extract mIoU values
    MIOU_1S=$(grep "mIoU.*at 1s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    MIOU_2S=$(grep "mIoU.*at 2s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)
    MIOU_3S=$(grep "mIoU.*at 3s" "$LOG_FILE" | grep -oP '[\d.]+' | tail -1)

    echo "=> mIoU@1s: $MIOU_1S, mIoU@2s: $MIOU_2S, mIoU@3s: $MIOU_3S"
    echo "=> IoU@1s: $IOU_1S, IoU@2s: $IOU_2S, IoU@3s: $IOU_3S"
    echo "$epoch,$MIOU_1S,$IOU_1S,$MIOU_2S,$IOU_2S,$MIOU_3S,$IOU_3S," >> "$SUMMARY_FILE"

    # Increment port to avoid conflicts
    MASTER_PORT=$((MASTER_PORT + 1))
    echo ""
done

echo "============================================"
echo "Evaluation complete!"
echo "============================================"
echo ""
echo "Summary:"
column -t -s',' "$SUMMARY_FILE"
