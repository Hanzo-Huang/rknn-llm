#!/usr/bin/env python3
"""Convenience conversion wrapper for the Rockchip-generated examples.

This file intentionally lives outside the generated example directories.  The
generated scripts are kept as the upstream reference; this wrapper is the
small, stable command line interface used by this repository.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LLM_EXPORT = ROOT / "examples" / "rkllm_api_demo" / "export"
VLM_EXPORT = ROOT / "examples" / "multimodal_model_demo" / "export"
VLM_DATA = ROOT / "examples" / "multimodal_model_demo" / "data"
PLATFORMS = ("RK3576", "RK3588")
DTYPES = (
    "fp",
    "w4a16",
    "w4a16_g32",
    "w4a16_g64",
    "w4a16_g128",
    "w8a8",
    "w8a8_g128",
    "w8a8_g256",
    "w8a8_g512",
)
DEFAULT_DTYPES = {"RK3576": "w4a16", "RK3588": "w8a8"}
VLM_MODELS = ("minicpm-v-2_6", "qwen2_5-vl-3b", "qwen3-vl", "qwen3.5", "smolvlm", "internvl3-1b", "deepseekocr")


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def context_length(value: str) -> int:
    result = positive_int(value)
    if result > 16_384:
        raise argparse.ArgumentTypeError("must not exceed 16384")
    if result % 32:
        raise argparse.ArgumentTypeError("must be a multiple of 32")
    return result


def unit_interval(value: str) -> float:
    result = float(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face LLM or VLM to Rockchip RKLLM/RKNN artifacts."
    )
    parser.add_argument("--kind", choices=("llm", "vlm"), default="llm", help="conversion type (default: llm)")
    parser.add_argument("-m", "--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("-p", "--platform", default="RK3588", help="target platform, RK3576, RK3588, or ALL (default: RK3588)")
    parser.add_argument("-q", "--dtype", choices=DTYPES, default=None, help="quantization type (default: w4a16 for RK3576, w8a8 for RK3588; with --platform ALL, omitted uses the supported platform matrix; fp disables quantization)")
    parser.add_argument("--dataset", default=None, help="RKLLM calibration dataset; generated when omitted for quantized conversion")
    parser.add_argument("-o", "--output", default=None, help="output .rkllm path; with ALL, used as the fan-out base path")
    parser.add_argument("--output-dir", default=None, help="directory for platform-specific VLM artifacts (default: output/<model>/<platform>)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="device used while loading the language model")
    parser.add_argument("--load-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--model-lora", default=None, help="optional LoRA model path")
    parser.add_argument("--custom-config", default=None, help="JSON file passed as RKLLM custom_config")
    parser.add_argument("-O", "--optimization-level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("-c", "--max-context", type=context_length, default=4096, help="maximum context length (a multiple of 32, up to 16384)")
    parser.add_argument("--num-npu-core", type=positive_int, default=None)
    parser.add_argument("--quantized-algorithm", choices=("normal", "grq"), default=None)
    parser.add_argument("--hybrid-rate", type=unit_interval, default=0)
    parser.add_argument("--extra-qparams", default=None, help="JSON file or JSON object passed as extra_qparams")
    parser.add_argument("--cuda-visible-devices", default=None, help="optional CUDA device mask, for example 0 or 1")
    parser.add_argument("--model-name", choices=VLM_MODELS, default=None, help="VLM vision wrapper name (auto-detected when omitted)")
    parser.add_argument("--model-type", choices=("qwen2vl", "qwen2.5vl", "qwen3vl", "qwen3.5"), default=None, help="used with --prepare-dataset")
    parser.add_argument("--prepare-dataset", action="store_true", help="generate VLM calibration data using the upstream helper")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--height", type=positive_int, default=448)
    parser.add_argument("--width", type=positive_int, default=448)
    parser.add_argument("--vision-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--vision-stage", choices=("all", "onnx", "rknn"), default="all", help="VLM vision artifact to generate (default: all)")
    parser.add_argument("--vision-only", action="store_true", help="for VLM, do not export the language RKLLM component")
    parser.add_argument("--skip-vision", action="store_true", help="for VLM, export only the RKLLM language component")
    parser.add_argument("--force", "--overwrite", dest="force", action="store_true", help="rebuild and overwrite existing artifacts")
    return parser


def platform_name(value: str) -> str:
    normalized = value.upper()
    if normalized not in (*PLATFORMS, "ALL"):
        raise SystemExit(f"Unsupported platform: {value}. Choose RK3576, RK3588, or ALL.")
    return normalized


def conversion_matrix(platform: str, dtype: str | None) -> list[tuple[str, str]]:
    """Return (platform, dtype) pairs for the requested conversion.

    Omitting ``--dtype`` with ``--platform ALL`` is the common batch workflow:
    one calibration dataset feeds these three builds.
    """
    if platform != "ALL":
        return [(platform, dtype or DEFAULT_DTYPES.get(platform, "w8a8"))]
    if dtype is not None:
        return [(target, dtype) for target in PLATFORMS]
    return [
        ("RK3588", "w8a8"),
        ("RK3576", "w4a16_g128"),
        ("RK3576", "w8a8"),
    ]


def core_count(platform: str, explicit: int | None) -> int:
    maximum = {"RK3576": 2, "RK3588": 3}.get(platform.upper())
    if maximum is None:
        raise SystemExit(f"Unsupported platform: {platform}")
    if explicit is not None:
        if explicit > maximum:
            raise SystemExit(f"{platform} supports at most {maximum} NPU cores")
        return explicit
    return maximum


def read_json(value: str | None) -> Any:
    if value is None:
        return None
    path = Path(value).expanduser()
    try:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SystemExit(f"Invalid JSON file: {path}") from error
    except OSError:
        # A JSON object can be longer than a valid filesystem path. In that
        # case it should be parsed as inline JSON below.
        pass
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Expected a JSON file or inline JSON object: {value!r}") from error


@contextmanager
def rkllm_model_view(model: Path):
    """Provide RKLLM a compatible view of legacy Qwen2.5-VL checkpoints.

    The original Qwen2.5-VL checkpoint stores the language-model fields at the
    top level.  RKLLM 1.3.0 unwraps the VLM as a text model and expects those
    fields under ``text_config`` instead.  Build a temporary symlink farm and
    add that compatibility view without changing the user's checkpoint.
    """
    if not model.is_dir():
        yield model
        return

    config_path = model / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        yield model
        return

    if config.get("model_type") != "qwen2_5_vl" or "text_config" in config:
        yield model
        return

    rope_scaling = config.get("rope_scaling")
    if not isinstance(rope_scaling, dict) or "mrope_section" not in rope_scaling:
        yield model
        return

    language_keys = (
        "attention_dropout", "bos_token_id", "eos_token_id", "hidden_act",
        "hidden_size", "initializer_range", "intermediate_size",
        "max_position_embeddings", "max_window_layers", "num_attention_heads",
        "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta",
        "sliding_window", "tie_word_embeddings", "torch_dtype", "use_cache",
        "use_sliding_window", "vocab_size",
    )
    text_config = {key: config[key] for key in language_keys if key in config}
    text_config.update({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2_5_vl_text",
        "rope_scaling": rope_scaling,
    })
    patched = dict(config)
    patched["text_config"] = text_config

    with tempfile.TemporaryDirectory(prefix="rkllm-qwen25vl-") as temp:
        staged = Path(temp) / model.name
        staged.mkdir()
        for source in model.iterdir():
            target = staged / source.name
            if source.name != "config.json":
                target.symlink_to(source, target_is_directory=source.is_dir())
        (staged / "config.json").write_text(
            json.dumps(patched, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Using temporary RKLLM compatibility view with text_config")
        yield staged


def make_vlm_dataset_paths_absolute(source: Path, destination: Path,
                                    sample_root: Path | None = None) -> None:
    """Copy a VLM calibration manifest without copying its sample directory."""
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid VLM calibration manifest: {source}") from error
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise SystemExit(f"VLM calibration manifest must be a JSON list of objects: {source}")
    sample_root = sample_root or source.parent
    for record in records:
        sample = record.get("sample")
        if sample and not Path(sample).is_absolute():
            record["sample"] = str((sample_root / sample).resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_calibration_dataset(dataset: Path) -> None:
    """Fail before loading a model when a quantization manifest is unusable."""
    if not dataset.is_file():
        raise SystemExit(f"Calibration dataset does not exist: {dataset}")
    try:
        records = json.loads(dataset.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Calibration dataset is not valid JSON: {dataset}") from error
    if not isinstance(records, list):
        raise SystemExit(f"Calibration dataset must be a JSON list: {dataset}")


def default_output(model: Path, dtype: str, platform: str) -> Path:
    suffix = ".rkllm"
    directory = ROOT / "output" / model.name / platform
    return directory / f"{model.name}_{platform}_{dtype}{suffix}"


def requested_output(model: Path, dtype: str, platform: str,
                     requested: str | None, multiple: bool) -> Path:
    """Resolve one output, including the legacy fan-out behavior for ALL."""
    if not requested:
        return default_output(model, dtype, platform)
    output = Path(requested).expanduser().resolve()
    if not multiple:
        return output
    stem = output.name[:-len(".rkllm")] if output.name.endswith(".rkllm") else output.name
    return output.parent / platform / f"{stem}_{platform}_{dtype}.rkllm"


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(str(part) for part in command))
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as error:
        if any("export_vision" in part for part in command):
            raise SystemExit("Vision export failed; see the traceback above.") from error
        raise


def export_rkllm(args: argparse.Namespace, model: Path, dataset: Path | None,
                 output: Path | None = None, platform_override: str | None = None,
                 dtype_override: str | None = None) -> Path:
    from rkllm.api import RKLLM

    platform = platform_override or platform_name(args.platform)
    dtype = dtype_override or args.dtype or DEFAULT_DTYPES.get(platform, "w8a8")
    output = output or (Path(args.output).expanduser().resolve() if args.output else default_output(model, dtype, platform))
    output.parent.mkdir(parents=True, exist_ok=True)
    if dataset is None and dtype != "fp":
        raise SystemExit("A calibration dataset is required for quantized conversion; pass --dataset.")

    load_config = read_json(args.custom_config)
    extra_qparams = read_json(args.extra_qparams)
    quantized = dtype != "fp"
    algorithm = args.quantized_algorithm or ("grq" if dtype.startswith("w4") else "normal")
    llm = RKLLM()
    with ExitStack() as stack:
        load_model = stack.enter_context(rkllm_model_view(model))
        print(f"Loading model: {model}")
        ret = llm.load_huggingface(
            model=str(load_model), model_lora=args.model_lora, device=args.device,
            dtype=args.load_dtype, custom_config=load_config, load_weight=True,
        )
        if ret != 0:
            raise RuntimeError(f"load_huggingface failed with status {ret}")
        # Older RKLLM releases validate quantized_dtype even when quantization is
        # disabled.  Keep the user-facing `fp` spelling while using a harmless
        # toolkit-compatible value internally.
        api_dtype = "w8a8" if dtype == "fp" else dtype
        ret = llm.build(
            do_quantization=quantized, optimization_level=args.optimization_level,
            quantized_dtype=api_dtype, quantized_algorithm=algorithm,
            target_platform=platform, num_npu_core=core_count(platform, args.num_npu_core),
            extra_qparams=extra_qparams, dataset=str(dataset) if dataset else None,
            hybrid_rate=args.hybrid_rate, max_context=args.max_context,
        )
        if ret != 0:
            raise RuntimeError(f"build failed with status {ret}")
        ret = llm.export_rkllm(str(output))
        if ret != 0:
            raise RuntimeError(f"export_rkllm failed with status {ret}")
    print(f"Wrote: {output}")
    return output


def infer_vlm_model_name(model: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = str(model).lower().replace("-", "_")
    aliases = (
        ("qwen3.5", "qwen3.5"), ("qwen3_5", "qwen3.5"),
        ("qwen3_vl", "qwen3-vl"), ("qwen2.5_vl", "qwen2_5-vl-3b"),
        ("qwen2_5_vl", "qwen2_5-vl-3b"), ("minicpm_v_2_6", "minicpm-v-2_6"),
        ("smolvlm", "smolvlm"), ("internvl3", "internvl3-1b"),
        ("deepseekocr", "deepseekocr"),
    )
    for marker, model_name in aliases:
        if marker in name:
            return model_name
    raise SystemExit("Could not detect VLM model type; pass --model-name explicitly.")


def infer_dataset_model_type(model: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = str(model).lower().replace("-", "_")
    if "qwen3.5" in name or "qwen3_5" in name:
        return "qwen3.5"
    if "qwen3_vl" in name:
        return "qwen3vl"
    if "qwen2.5_vl" in name or "qwen2_5_vl" in name:
        return "qwen2.5vl"
    raise SystemExit("Could not detect VLM calibration model type; pass --model-type explicitly.")


def export_vlm_vision(args: argparse.Namespace, model: Path, output_dir: Path,
                      platform: str, export_onnx: bool | None = None) -> tuple[Path, Path]:
    model_name = infer_vlm_model_name(model, args.model_name)
    onnx = VLM_EXPORT / "onnx" / f"{model_name}_vision.onnx"
    rknn = VLM_EXPORT / "rknn" / f"{model_name}_vision_{platform.lower()}.rknn"
    launcher = ROOT / "tools" / "run_vision.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir.parent
    onnx_dir.mkdir(parents=True, exist_ok=True)
    copied_onnx = onnx_dir / f"{model.name}_vision.onnx"
    copied_rknn = output_dir / f"{model.name}_vision_{platform}.rknn"

    run_onnx = args.vision_stage in ("all", "onnx") and (args.force or not copied_onnx.exists())
    if export_onnx is not None:
        run_onnx = run_onnx and export_onnx
    run_rknn = args.vision_stage in ("all", "rknn") and (args.force or not copied_rknn.exists())
    if run_onnx:
        run([sys.executable, str(launcher), "--stage", "onnx", "--script", str(VLM_EXPORT / "export_vision.py"),
         "--path", str(model), "--model_name", model_name,
         "--batch_size", str(args.batch_size), "--height", str(args.height), "--width", str(args.width), "--device", args.vision_device], VLM_EXPORT)
    elif run_rknn and not onnx.exists():
        if copied_onnx.exists():
            onnx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(copied_onnx, onnx)
        else:
            raise SystemExit(f"Cannot build RKNN: source ONNX does not exist: {onnx}")
    if run_rknn:
        run([sys.executable, str(launcher), "--stage", "rknn", "--script", str(VLM_EXPORT / "export_vision_rknn.py"),
         "--path", str(onnx), "--model_name", model_name,
         "--target-platform", platform.lower(), "--batch_size", str(args.batch_size), "--height", str(args.height), "--width", str(args.width)], VLM_EXPORT)
    if run_onnx:
        if not onnx.is_file():
            raise SystemExit(f"Vision exporter did not create ONNX output: {onnx}")
        shutil.copy2(onnx, copied_onnx)
        print(f"Wrote: {copied_onnx}")
    if run_rknn:
        if not rknn.is_file():
            raise SystemExit(f"Vision exporter did not create RKNN output: {rknn}")
        shutil.copy2(rknn, copied_rknn)
        print(f"Wrote: {copied_rknn}")
    elif args.vision_stage in ("all", "rknn") and copied_rknn.exists():
        print(f"Skipping existing: {copied_rknn}")
    if not run_onnx and args.vision_stage in ("all", "onnx") and copied_onnx.exists():
        print(f"Skipping existing: {copied_onnx}")
    if args.vision_stage in ("all", "onnx") and not copied_onnx.is_file():
        raise SystemExit(f"ONNX artifact is unavailable: {copied_onnx}")
    if args.vision_stage in ("all", "rknn") and not copied_rknn.is_file():
        raise SystemExit(f"RKNN artifact is unavailable: {copied_rknn}")
    return copied_onnx, copied_rknn


def main() -> int:
    args = build_parser().parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    model = Path(args.model).expanduser().resolve() if Path(args.model).exists() else Path(args.model)
    platform = platform_name(args.platform)
    matrix = conversion_matrix(platform, args.dtype)
    language_outputs = [
        requested_output(model, dtype, target_platform, args.output, len(matrix) > 1)
        for target_platform, dtype in matrix
    ]
    language_pending = not args.vision_only and (
        args.force or any(not output.exists() for output in language_outputs)
    )
    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else None
    model_output_dir = ROOT / "output" / model.name
    if args.kind == "llm" and language_pending and dataset is None:
        dataset = model_output_dir / f"{model.name}_data_quant.json"
        if args.dtype != "fp" and not dataset.exists():
            dataset.parent.mkdir(parents=True, exist_ok=True)
            run([sys.executable, str(LLM_EXPORT / "generate_data_quant.py"), "-m", str(model), "-o", str(dataset)], LLM_EXPORT)
        elif args.dtype != "fp" and dataset.exists():
            print(f"Skipping existing calibration dataset: {dataset}")
    if args.kind == "vlm" and language_pending and dataset is None:
        dataset = model_output_dir / f"{model.name}_data_quant.json"
    if args.kind == "vlm" and args.prepare_dataset and dataset is None:
        dataset = model_output_dir / f"{model.name}_data_quant.json"
    if args.kind == "vlm" and (args.prepare_dataset or (language_pending and args.dtype != "fp")) and dataset is not None and dataset.exists() and not args.force:
        print(f"Skipping existing calibration dataset: {dataset}")
    elif args.kind == "vlm" and (args.prepare_dataset or (language_pending and args.dtype != "fp")) and dataset is not None:
        model_type = infer_dataset_model_type(model, args.model_type)
        run([sys.executable, str(VLM_DATA / "make_input_embeds_for_quantize.py"), "--path", str(model), "--model_type", model_type], VLM_DATA.parent)
        generated = VLM_DATA / "llm_inputs.json"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        make_vlm_dataset_paths_absolute(generated, dataset, VLM_DATA)
    elif args.kind == "vlm" and dataset is not None and dataset.parent == model_output_dir:
        # Repair manifests created by an earlier wrapper version. Keep the
        # sample files in the example data directory, outside the output tree.
        make_vlm_dataset_paths_absolute(dataset, dataset, VLM_DATA)
    if language_pending and any(dtype != "fp" for _, dtype in matrix):
        if dataset is None:
            raise SystemExit("A calibration dataset is required for quantized conversion; pass --dataset.")
        validate_calibration_dataset(dataset)
    if args.kind == "vlm" and not args.skip_vision:
        custom_vision_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
        vision_platforms = list(dict.fromkeys(target for target, _ in matrix))
        for index, vision_platform in enumerate(vision_platforms):
            vision_dir = (custom_vision_dir / vision_platform if custom_vision_dir and platform == "ALL"
                          else custom_vision_dir if custom_vision_dir else model_output_dir / vision_platform)
            export_vlm_vision(args, model, vision_dir, vision_platform,
                              export_onnx=(index == 0 if platform == "ALL" else None))
    if args.kind == "vlm" and args.vision_only:
        return 0
    for target_platform, dtype in matrix:
        output = requested_output(model, dtype, target_platform, args.output, len(matrix) > 1)
        if output.exists() and not args.force:
            print(f"Skipping existing: {output}")
            continue
        export_rkllm(args, model, dataset, output,
                     platform_override=target_platform, dtype_override=dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
