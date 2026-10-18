#!/usr/bin/env python3
"""RKNN encoder export using the SDK 1.3.1 input and normalization settings.

Keep convenience arguments and checked status handling outside upstream files.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--target-platform', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--width', type=int, default=448)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    import numpy as np
    from rknn.api import RKNN

    name = args.model_name
    audio = name == 'gemma-4-audio'
    config = {'target_platform': args.target_platform}
    load = {}
    if not audio:
        mean, std = [0.5] * 3, [0.5] * 3
        if name in ('qwen2-vl', 'qwen2_5-vl-3b'):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        elif name == 'internvl3-1b':
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif name == 'gemma-4':
            mean, std = [0] * 3, [1] * 3
        config.update(mean_values=[[x * 255 for x in mean]],
                      std_values=[[x * 255 for x in std]],
                      disable_rules=['convert_rs_add_rs_to_rs_gather_elements'] if name == 'deepseekocr' else [])
        if name in ('qwen2_5-vl-3b', 'qwen3-vl', 'qwen3.5'):
            patch = 14 if name == 'qwen2_5-vl-3b' else 16
            load = dict(inputs=['pixel', 'grid_thw'],
                        input_size_list=[[args.batch_size, 3, args.height, args.width], [1, 3]],
                        input_initial_val=[None, np.array([[(args.batch_size + 1) // 2,
                                                           args.height // patch, args.width // patch]], dtype=np.int64)])
            if name == 'qwen2_5-vl-3b':
                config['op_target'] = {'/vpm/patch_embed/proj/Conv_output_0_conv_tp_sw': 'cpu'}
        elif name == 'gemma-4':
            height, width = ((value + 47) // 48 * 48 for value in (args.height, args.width))
            y, x = np.meshgrid(np.arange(height // 16), np.arange(width // 16), indexing='ij')
            positions = np.stack([x, y], axis=-1).reshape(1, -1, 2).astype(np.int64)
            load = dict(inputs=['pixel', 'pixel_position_ids'],
                        input_size_list=[[1, 3, height, width], [1, height // 16 * (width // 16), 2]],
                        input_initial_val=[None, positions])
        elif name != 'qwen2-vl':
            load = dict(inputs=['pixel'], input_size_list=[[args.batch_size, 3, args.height, args.width]])
        # The legacy Qwen2 exporter fixes shape/positions and lets ONNX name its
        # input. Leave inputs unspecified so RKNN reads that name from the graph.

    output = args.output or Path('rknn') / f'{args.path.stem}_{args.target_platform}.rknn'
    output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        for operation, kwargs in (
            ('config', config),
            ('load_onnx', dict(model=str(args.path), **load)),
            ('build', dict(do_quantization=False, dataset=None)),
            ('export_rknn', dict(export_path=str(output))),
        ):
            status = getattr(rknn, operation)(**kwargs)
            if status not in (None, 0):
                raise RuntimeError(f'RKNN {operation} failed with status {status}')
    finally:
        rknn.release()


if __name__ == '__main__':
    main()
