#!/usr/bin/env bash
set -uo pipefail

work_dir="${1:-/data/jxy/projects/work_dirs/our-nusc-base}"
archive_dir="${2:-${work_dir}/all_epoch_pths}"
interval="${3:-30}"

mkdir -p "$archive_dir"

while true; do
  find "$work_dir" -maxdepth 1 -type f -name 'epoch_*.pth' -print0 |
    while IFS= read -r -d '' ckpt; do
      ln -f "$ckpt" "$archive_dir/$(basename "$ckpt")"
    done
  sleep "$interval"
done
