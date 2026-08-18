#!/usr/bin/env python3
"""Run either Rockchip VLM vision export stage with compatibility shims."""

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path


def patch_transformers() -> None:
    import transformers

    for class_name in ("Qwen2_5_VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None and not hasattr(model_class, "visual"):
            # Rockchip v1.3.0 accesses vlm.visual; newer Transformers places
            # the same module at vlm.model.visual.
            setattr(model_class, "visual", property(lambda instance: instance.model.visual))


def patch_onnx() -> None:
    import onnx

    if hasattr(onnx, "mapping") or not hasattr(onnx, "_mapping"):
        return
    tensor_type_map = getattr(onnx._mapping, "TENSOR_TYPE_MAP", {})
    tensor_type_to_np_type = {
        key: value.np_dtype for key, value in tensor_type_map.items()
    }
    np_type_to_tensor_type = {}
    for key, np_dtype in tensor_type_to_np_type.items():
        np_type_to_tensor_type[np_dtype] = key
        if hasattr(np_dtype, "type"):
            np_type_to_tensor_type[np_dtype.type] = key
    onnx.mapping = types.SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE=tensor_type_to_np_type,
        NP_TYPE_TO_TENSOR_TYPE=np_type_to_tensor_type,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Rockchip VLM vision conversion stage")
    parser.add_argument("--stage", choices=("onnx", "rknn"), required=True)
    parser.add_argument("--script", required=True, help="original Rockchip vision script")
    args, script_args = parser.parse_known_args()
    script = Path(args.script).resolve()
    sys.argv = [str(script), *script_args]

    if args.stage == "onnx":
        patch_transformers()
        source = script.read_text(encoding="utf-8")
        if "qwen2_5-vl-3b" in sys.argv:
            source = source.replace(
                "return self.vpm(flatten_patches, grid_thw)",
                "return self.vpm(flatten_patches, grid_thw).pooler_output",
            )
        if "deepseekocr" in sys.argv:
            marker = "        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)\n        model = deepseekocr_vision(model.model)"
            replacement = '''        # Rebuild deterministic position IDs omitted from the checkpoint.
        for vision_module in model.modules():
            position_ids = getattr(vision_module, "position_ids", None)
            num_positions = getattr(vision_module, "num_positions", None)
            if isinstance(position_ids, torch.Tensor) and isinstance(num_positions, int):
                vision_module.position_ids = torch.arange(
                    num_positions, device=position_ids.device
                ).expand((1, -1))
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        model = deepseekocr_vision(model.model)'''
            if marker not in source:
                raise SystemExit(
                    "The upstream DeepSeek-OCR exporter changed; cannot apply "
                    "the compatibility patch."
                )
            source = source.replace(marker, replacement, 1)
        exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
    else:
        patch_onnx()
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
