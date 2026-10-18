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

    for class_name in ("Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None and not hasattr(model_class, "visual"):
            # Rockchip v1.3.0 accesses vlm.visual; newer Transformers places
            # the same module at vlm.model.visual.
            setattr(model_class, "visual", property(lambda instance: instance.model.visual))


def patch_gemma4_export() -> None:
    """Normalize scalar shape lengths during fixed-size Gemma4 ONNX tracing.

    Transformers 5.8 treats tensor lengths as cache-position vectors, but JIT
    also turns ordinary shape integers into zero-dimensional tensors.
    """
    import functools
    import inspect
    import torch
    import transformers.masking_utils as masks

    original = masks.sdpa_mask
    signature = inspect.signature(original)

    @functools.wraps(original)
    def fixed_shape_mask(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if torch.jit.is_tracing():
            for name in ("q_length", "kv_length", "q_offset", "kv_offset"):
                value = bound.arguments.get(name)
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    bound.arguments[name] = int(value)
        return original(*bound.args, **bound.kwargs)

    masks.sdpa_mask = fixed_shape_mask


def stable_rms_norm(self, hidden_states):
    """Equivalent RMS normalization whose square stays in FP16 range.

    RKNN lowers the vision graph to FP16, including the upstream FP32 cast.
    Gemma4 residuals can exceed 500, overflowing x**2 at layer 6. Per-token
    scaling bounds the square to one and retains epsilon in the scaled domain.
    """
    import torch

    root_eps = self.eps ** 0.5
    scale = hidden_states.abs().amax(-1, keepdim=True).clamp_min(root_eps)
    scaled = hidden_states / scale
    variance = scaled.pow(2).mean(-1, keepdim=True) + (root_eps / scale).pow(2)
    return scaled * torch.pow(variance, -0.5)


def patch_gemma4_vision_model(module) -> None:
    """Avoid RKNN's unsupported boolean GatherND for a fixed unpadded grid.

    export_vision_gemma4 uses a full rectangular grid with no (-1, -1) padding.
    Every pooled token is valid, so boolean selection is exactly a reshape.
    """
    import ast

    module.Gemma4RMSNorm._norm = stable_rms_norm
    tree = ast.parse(Path(module.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Gemma4VisionModel")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    forward.decorator_list = []
    source = ast.unparse(forward)
    original = "hidden_states = hidden_states[pooler_mask]"
    if original not in source:
        raise RuntimeError("Gemma4 vision pooling changed; review the fixed-grid export shim")
    source = source.replace(original, "hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])")
    namespace = dict(vars(module))
    exec(compile(source, module.__file__, "exec"), namespace)
    module.Gemma4VisionModel.forward = namespace["forward"]


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
        if script.name == "export_vision_gemma4.py":
            patch_gemma4_export()
        source = script.read_text(encoding="utf-8")
        if "qwen2_5-vl-3b" in sys.argv:
            source = source.replace(
                "return self.vpm(flatten_patches, grid_thw)",
                "return self.vpm(flatten_patches, grid_thw).pooler_output",
            )
        if script.name == "export_vision_gemma4.py":
            source = source.replace(
                "    modeling_spec.loader.exec_module(modeling_module)",
                "    modeling_spec.loader.exec_module(modeling_module)\n    patch_gemma4_vision_model(modeling_module)",
            )
        exec(compile(source, str(script), "exec"), {
            "__name__": "__main__", "__file__": str(script),
            "patch_gemma4_vision_model": patch_gemma4_vision_model,
        })
    else:
        patch_onnx()
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
