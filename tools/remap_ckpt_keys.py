#!/usr/bin/env python
"""Remap old checkpoint keys (pose_aligner/memory_attn/memory_fusion) to
new STAC-QM naming (query_memory.aligner/attention/fusion)."""

import argparse
import torch

KEY_MAP = [
    ('pose_aligner.', 'query_memory.aligner.'),
    ('memory_attn.', 'query_memory.attention.'),
    ('memory_fusion.', 'query_memory.fusion.'),
]


def remap_state_dict(state_dict):
    remapped = {}
    remapped_count = 0
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in KEY_MAP:
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                remapped_count += 1
                break
        remapped[new_key] = value
    return remapped, remapped_count


def main():
    parser = argparse.ArgumentParser(description='Remap old checkpoint keys to STAC-QM naming')
    parser.add_argument('input', help='Input checkpoint path')
    parser.add_argument('--output', '-o', help='Output path (default: input with _remapped suffix)')
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    remapped, count = remap_state_dict(state_dict)

    if 'state_dict' in ckpt:
        ckpt['state_dict'] = remapped
    else:
        ckpt = remapped

    out_path = args.output or args.input.replace('.pth', '_remapped.pth')
    torch.save(ckpt, out_path)
    print(f'Remapped {count} keys → {out_path}')


if __name__ == '__main__':
    main()
