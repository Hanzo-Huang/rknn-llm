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
import re
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
# RKLLM 1.3.1 SDK manual, section 3.1.3, and rkllm/api/rkllm.py.
W4 = ("w4a16", "w4a16_g32", "w4a16_g64", "w4a16_g128")
PLATFORM_DTYPES = {
    "RK3576": (*W4, "w8a8"),
    "RK3588": ("w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512"),
    "RK3562": ("w8a8", *W4[1:], "w4a8_g32"),
    "RV1126B": (*W4, "w8a8"),
    "CUDA": ("w4a16_g32",),  # Jetson; explicit target, excluded from ALL.
}
PLATFORMS = tuple(target for target in PLATFORM_DTYPES if target != "CUDA")
DTYPES = ("fp", *dict.fromkeys(dtype for values in PLATFORM_DTYPES.values() for dtype in values), "all")
DEFAULT_DTYPES = {"RK3576": "w4a16_g128", "RK3588": "w8a8",
                  "RK3562": "w8a8", "RV1126B": "w4a16_g128", "CUDA": "w4a16_g32"}
VLM_MODELS = ("minicpm-v-2_6", "qwen2-vl", "qwen2_5-vl-3b", "qwen3-vl", "qwen3.5", "smolvlm", "internvl3-1b", "deepseekocr", "gemma-4")


def quantization_algorithm(value: str) -> str:
    if not re.fullmatch(r"(?:normal|grq)(?:_r(?:[1-9]|[1-5][0-9]|6[0-4]))?|none", value):
        raise argparse.ArgumentTypeError("choose none, normal, grq, normal_r1..64, or grq_r1..64")
    return value


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
    if not 0 <= result < 1:
        raise argparse.ArgumentTypeError("must be at least 0 and less than 1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face LLM or VLM to Rockchip RKLLM/RKNN artifacts."
    )
    parser.add_argument("--kind", choices=("llm", "vlm"), default="llm", help="conversion type (default: llm)")
    parser.add_argument("-m", "--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--format", choices=("auto", "huggingface", "gguf"), default="auto", help="input format; auto detects .gguf files")
    parser.add_argument("-p", "--platform", default="RK3588", help="target: RK3576, RK3588, RK3562, RV1126B, CUDA (Jetson), or ALL Rockchip targets")
    parser.add_argument("-q", "--dtype", choices=DTYPES, default=None, help="quantization type; all builds every supported quantized dtype per target; fp disables quantization")
    parser.add_argument("--dataset", default=None, help="RKLLM calibration dataset; generated when omitted for quantized conversion")
    parser.add_argument("-o", "--output", default=None, help="output .rkllm path; with ALL, used as the fan-out base path")
    parser.add_argument("--output-dir", default=None, help="common output directory for calibration and model artifacts (default: output/<model>)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="device used while loading the language model")
    parser.add_argument("--load-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--model-lora", default=None, help="optional LoRA model path")
    parser.add_argument("--custom-config", default=None, help="JSON file passed as RKLLM custom_config")
    parser.add_argument("-O", "--optimization-level", type=int, choices=(0, 1), default=1)
    parser.add_argument("-c", "--max-context", type=context_length, default=4096, help="maximum context length (a multiple of 32, up to 16384)")
    parser.add_argument("--num-npu-core", type=positive_int, default=None)
    parser.add_argument("--quantized-algorithm", type=quantization_algorithm, default=None)
    parser.add_argument("--hybrid-rate", type=unit_interval, default=0)
    parser.add_argument("--extra-qparams", default=None, help="cached .qparams path, JSON file, or inline JSON object")
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
    parser.add_argument("--audio", action="store_true", help="also export the Gemma4 audio encoder (with --kind vlm)")
    parser.add_argument("--audio-only", action="store_true", help="export only the Gemma4 audio encoder")
    parser.add_argument("--audio-stage", choices=("all", "onnx", "rknn"), default="all")
    parser.add_argument("--no-export-tokenizer", action="store_true")
    parser.add_argument("--no-export-embedding", action="store_true")
    parser.add_argument("--force", "--overwrite", dest="force", action="store_true", help="rebuild and overwrite existing artifacts")
    return parser


def platform_name(value: str) -> str:
    normalized = value.upper()
    if normalized not in (*PLATFORM_DTYPES, "ALL"):
        raise SystemExit(f"Unsupported platform: {value}. Choose {', '.join(PLATFORM_DTYPES)}, or ALL.")
    return normalized


def conversion_matrix(platform: str, dtype: str | None) -> list[tuple[str, str]]:
    targets = PLATFORMS if platform == "ALL" else (platform,)
    result = []
    for target in targets:
        if dtype == "all":
            dtypes = PLATFORM_DTYPES[target]
        elif dtype is not None:
            dtypes = (dtype,)
        elif platform == "ALL" and target != "RK3588":
            dtypes = ("w4a16_g128", "w8a8")
        else:
            dtypes = (DEFAULT_DTYPES[target],)
        for quantization in dtypes:
            if quantization != "fp" and quantization not in PLATFORM_DTYPES[target]:
                raise SystemExit(f"{target} does not support {quantization}; choose {', '.join(PLATFORM_DTYPES[target])}, or fp")
            if target == "CUDA" and quantization == "fp":
                raise SystemExit("CUDA supports only w4a16_g32")
            result.append((target, quantization))
    return result


def core_count(platform: str, explicit: int | None) -> int:
    maximum = {"RK3576": 2, "RK3588": 3, "RK3562": 1, "RV1126B": 1, "CUDA": 1}[platform.upper()]
    if explicit is not None:
        if not 1 <= explicit <= maximum:
            raise SystemExit(f"{platform} supports 1..{maximum} NPU cores")
        return explicit
    return maximum


def input_format(args: argparse.Namespace) -> str:
    return ("gguf" if args.model.lower().endswith(".gguf") else "huggingface") if args.format == "auto" else args.format


def qparams_value(value: str | None) -> Any:
    if value and Path(value).expanduser().suffix == ".qparams":
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Quantization cache does not exist: {path}")
        return str(path)
    return read_json(value)


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
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise SystemExit(f"Calibration dataset must be a nonempty JSON list of objects: {dataset}")
    for record in records:
        if "sample" in record:
            sample = Path(record["sample"])
            if not sample.is_absolute():
                sample = dataset.parent / sample
            if not sample.is_file():
                raise SystemExit(f"Calibration sample does not exist: {sample}")
        elif not isinstance(record.get("input"), str):
            raise SystemExit(f"Calibration record must contain input text or a sample path: {dataset}")


def default_output(model: Path, dtype: str, platform: str,
                   output_dir: Path | None = None) -> Path:
    suffix = ".rkllm"
    directory = (output_dir or model_output_dir(model)) / platform
    return directory / f"{model.name}_{platform}_{dtype}{suffix}"


def model_output_dir(model: Path) -> Path:
    """Return the default output directory for a local model.

    Preserve nested directories below the repository's ``models`` directory
    so, for example, ``models/Qwen3/Qwen3-4B`` maps to
    ``output/Qwen3/Qwen3-4B``. Model IDs and local models outside this
    repository continue to use a directory named after the model itself.
    """
    models_root = ROOT / "models"
    try:
        relative_model = model.relative_to(models_root)
    except ValueError:
        relative_model = Path(model.name)
    return ROOT / "output" / relative_model


def requested_output(model: Path, dtype: str, platform: str,
                     requested: str | None, multiple: bool,
                     output_dir: Path | None = None) -> Path:
    """Resolve one output, including the legacy fan-out behavior for ALL."""
    if not requested:
        return default_output(model, dtype, platform, output_dir)
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
    extra_qparams = qparams_value(args.extra_qparams)
    quantized = dtype != "fp"
    algorithm = args.quantized_algorithm or ("grq" if dtype.startswith("w4a16") else "normal")
    llm = RKLLM()
    with ExitStack() as stack:
        load_model = stack.enter_context(rkllm_model_view(model))
        print(f"Loading model: {model}")
        if input_format(args) == "gguf":
            ret = llm.load_gguf(model=str(load_model))
            operation = "load_gguf"
        else:
            ret = llm.load_huggingface(
                model=str(load_model), model_lora=args.model_lora, device=args.device,
                dtype=args.load_dtype, custom_config=load_config, load_weight=True,
            )
            operation = "load_huggingface"
        if ret != 0:
            raise RuntimeError(f"{operation} failed with status {ret}")
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
        ret = llm.export_rkllm(str(output), export_tokenizer=not args.no_export_tokenizer,
                               export_embedding=not args.no_export_embedding)
        if ret != 0:
            raise RuntimeError(f"export_rkllm failed with status {ret}")
    print(f"Wrote: {output}")
    return output


def infer_vlm_model_name(model: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = str(model).lower().replace("-", "_")
    aliases = (
        ("gemma_4", "gemma-4"), ("gemma4", "gemma-4"),
        ("qwen2_vl", "qwen2-vl"),
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
    if "qwen2_vl" in name:
        return "qwen2vl"
    if "qwen3.5" in name or "qwen3_5" in name:
        return "qwen3.5"
    if "qwen3_vl" in name:
        return "qwen3vl"
    if "qwen2.5_vl" in name or "qwen2_5_vl" in name:
        return "qwen2.5vl"
    raise SystemExit("No automatic calibration helper for this VLM. Pass --dataset with a prepared multimodal manifest; --model-type selects Qwen helpers only.")


def copy_onnx(source: Path, destination: Path) -> None:
    """Copy a graph together with external tensor files (large vision models)."""
    import onnx

    graph = onnx.load(str(source), load_external_data=False)
    locations = set()

    def collect(message):
        # Walk protobuf fields to include tensors inside subgraphs/attributes.
        if message.DESCRIPTOR.full_name == "onnx.TensorProto":
            locations.update(entry.value for entry in message.external_data if entry.key == "location")
        for field, value in message.ListFields():
            if field.message_type is not None:
                if field.label == field.LABEL_REPEATED:
                    for child in value:
                        collect(child)
                else:
                    collect(value)

    collect(graph)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for location in locations:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"Unsafe ONNX external data location: {location}")
        target = destination.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if (source.parent / relative).resolve() != target.resolve():
            shutil.copy2(source.parent / relative, target)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def export_vlm_vision(args: argparse.Namespace, model: Path, output_dir: Path,
                      platform: str, export_onnx: bool | None = None) -> tuple[Path, Path]:
    model_name = infer_vlm_model_name(model, args.model_name)
    onnx = output_dir.parent / f"{model.name}_vision.onnx"
    rknn = output_dir / f"{model.name}_vision_{platform}.rknn"
    launcher = ROOT / "tools" / "run_vision.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_onnx = args.vision_stage in ("all", "onnx") and (args.force or not onnx.is_file())
    if export_onnx is not None:
        run_onnx = run_onnx and export_onnx
    run_rknn = args.vision_stage in ("all", "rknn") and (args.force or not rknn.is_file())
    if run_onnx:
        dimensions = ["--batch_size", str(args.batch_size), "--height", str(args.height), "--width", str(args.width)]
        if model_name == "gemma-4":
            run([sys.executable, str(VLM_EXPORT / "export_vision_gemma4.py"),
                 "--model_path", str(model), "--export_vision_path", str(onnx), *dimensions,
                 "--device", args.vision_device], VLM_EXPORT)
        elif model_name == "qwen2-vl":
            for step in (1, 2):
                run([sys.executable, str(launcher), "--stage", "onnx", "--script", str(VLM_EXPORT / "export_vision_qwen2.py"),
                     "--step", str(step), "--path", str(model), "--savepath", str(onnx),
                     "--batch", str(args.batch_size), "--height", str(args.height), "--width", str(args.width)], VLM_EXPORT)
        else:
            generated = VLM_EXPORT / "onnx" / f"{model_name}_vision.onnx"
            run([sys.executable, str(launcher), "--stage", "onnx", "--script", str(VLM_EXPORT / "export_vision.py"),
                 "--path", str(model), "--model_name", model_name, *dimensions, "--device", args.vision_device], VLM_EXPORT)
            copy_onnx(generated, onnx)
    if args.vision_stage in ("all", "onnx") or run_rknn:
        if not onnx.is_file():
            raise SystemExit(f"Vision ONNX is unavailable: {onnx}; run --vision-stage onnx first")
    if run_rknn:
        run([sys.executable, str(launcher), "--stage", "rknn", "--script", str(ROOT / "tools" / "export_encoder_rknn.py"),
             "--path", str(onnx), "--model_name", model_name, "--output", str(rknn),
             "--target-platform", platform.lower(), "--batch_size", str(args.batch_size),
             "--height", str(args.height), "--width", str(args.width)], VLM_EXPORT)
    if args.vision_stage in ("all", "rknn") and not rknn.is_file():
        raise SystemExit(f"Vision RKNN is unavailable: {rknn}")
    return onnx, rknn


def export_audio(args: argparse.Namespace, model: Path, output_root: Path, platform: str,
                 export_onnx: bool) -> None:
    onnx = output_root / f"{model.name}_audio.onnx"
    rknn = output_root / platform / f"{model.name}_audio_{platform}.rknn"
    if export_onnx and args.audio_stage in ("all", "onnx") and (args.force or not onnx.is_file()):
        run([sys.executable, str(VLM_EXPORT / "export_audio_gemma4.py"),
             "--model_path", str(model), "--export_audio_path", str(onnx)], VLM_EXPORT)
        if not onnx.is_file():
            raise SystemExit(f"Audio exporter did not create: {onnx}")
    if args.audio_stage in ("all", "rknn") and (args.force or not rknn.is_file()):
        if not onnx.is_file():
            raise SystemExit(f"Cannot build audio RKNN: source ONNX does not exist: {onnx}")
        rknn.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(ROOT / "tools" / "export_encoder_rknn.py"),
             "--path", str(onnx), "--model_name", "gemma-4-audio",
             "--target-platform", platform.lower(), "--output", str(rknn)], VLM_EXPORT)


def validate_options(args: argparse.Namespace, model: Path, matrix: list[tuple[str, str]]) -> None:
    if args.vision_only and args.skip_vision:
        raise SystemExit("--vision-only cannot be combined with --skip-vision")
    if args.kind != "vlm" and any((args.audio, args.audio_only, args.vision_only, args.skip_vision, args.prepare_dataset)):
        raise SystemExit("vision/audio and multimodal calibration options require --kind vlm")
    if args.audio_only and (args.vision_only or args.prepare_dataset):
        raise SystemExit("--audio-only cannot be combined with --vision-only or --prepare-dataset")
    if args.kind == "vlm":
        if any(target == "CUDA" for target, _ in matrix):
            raise SystemExit("VLM encoder conversion requires a Rockchip target")
        if not args.skip_vision and not args.audio_only:
            name = infer_vlm_model_name(model, args.model_name)
            if name == "gemma-4" and args.batch_size != 1:
                raise SystemExit("The upstream Gemma4 vision exporter supports batch size 1")
            alignment = 28 if name in ("qwen2-vl", "qwen2_5-vl-3b") else 32 if name in ("qwen3-vl", "qwen3.5") else 1
            if args.height % alignment or args.width % alignment:
                raise SystemExit(f"{name} image height and width must be multiples of {alignment}")
        if args.audio or args.audio_only:
            if infer_vlm_model_name(model, args.model_name) != "gemma-4":
                raise SystemExit("The SDK provides an audio exporter only for Gemma4")
    if input_format(args) == "gguf":
        if args.kind != "llm" or args.model_lora or args.custom_config:
            raise SystemExit("GGUF input requires --kind llm and does not accept Hugging Face LoRA/custom-config options")
        if not model.is_file():
            raise SystemExit("GGUF input must be a local .gguf file")
    for target, dtype in matrix:
        core_count(target, args.num_npu_core)
        algorithm = args.quantized_algorithm or ("grq" if dtype.startswith("w4a16") else "normal")
        if algorithm.startswith("grq") and dtype != "fp":
            if not dtype.startswith("w4a16"):
                raise SystemExit("GRQ supports only w4a16 and grouped w4a16")
            if args.device != "cuda":
                raise SystemExit("GRQ requires a CUDA GPU; use --device cuda or --quantized-algorithm normal")
        if args.model_lora and "_r" in algorithm:
            raise SystemExit("LoRA correction algorithms cannot be combined with --model-lora")


def main() -> int:
    args = build_parser().parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    model = Path(args.model).expanduser()
    if model.exists():
        model = model.resolve()
    platform = platform_name(args.platform)
    matrix = conversion_matrix(platform, args.dtype)
    validate_options(args, model, matrix)
    read_json(args.custom_config)
    qparams_value(args.extra_qparams)
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_output_dir(model)
    outputs = [requested_output(model, dtype, target, args.output, len(matrix) > 1, output_root)
               for target, dtype in matrix]
    for output in outputs:
        if output.is_file() and not args.force and not (args.vision_only or args.audio_only):
            print(f"Skipping existing: {output}")
    pending = [(target, dtype, output) for (target, dtype), output in zip(matrix, outputs)
               if args.force or not output.is_file()]
    if args.vision_only or args.audio_only:
        pending = []
    needs_calibration = any(dtype != "fp" for _, dtype, _ in pending)
    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else None
    if needs_calibration or args.prepare_dataset:
        if dataset is not None:
            # Explicit manifests are user input: never overwrite or regenerate them.
            validate_calibration_dataset(dataset)
        else:
            dataset = output_root / f"{model.name}_data_quant.json"
            if not dataset.is_file() or args.prepare_dataset or args.force:
                if input_format(args) == "gguf":
                    raise SystemExit("Pass --dataset for quantized GGUF conversion; the text calibration helper requires a Hugging Face model")
                if args.kind == "llm":
                    dataset.parent.mkdir(parents=True, exist_ok=True)
                    run([sys.executable, str(LLM_EXPORT / "generate_data_quant.py"), "-m", str(model), "-o", str(dataset)], LLM_EXPORT)
                else:
                    model_type = infer_dataset_model_type(model, args.model_type)
                    run([sys.executable, str(VLM_DATA / "make_input_embeds_for_quantize.py"), "--path", str(model), "--model_type", model_type], VLM_DATA.parent)
                    sample_root = output_root / "calibration"
                    shutil.copytree(VLM_DATA / "llm_inputs", sample_root / "llm_inputs", dirs_exist_ok=True)
                    make_vlm_dataset_paths_absolute(VLM_DATA / "llm_inputs.json", dataset, sample_root)
            validate_calibration_dataset(dataset)
        if args.kind == "vlm":
            # Resolve relative sample paths against the supplied manifest, not cwd.
            resolved = output_root / f"{model.name}_resolved_calibration.json"
            make_vlm_dataset_paths_absolute(dataset, resolved)
            dataset = resolved
    vision_platforms = list(dict.fromkeys(target for target, _ in matrix))
    if args.kind == "vlm":
        for index, target in enumerate(vision_platforms):
            if not args.skip_vision and not args.audio_only:
                export_vlm_vision(args, model, output_root / target, target, export_onnx=(index == 0))
            if args.audio or args.audio_only:
                export_audio(args, model, output_root, target, export_onnx=(index == 0))
    for target, dtype, output in pending:
        export_rkllm(args, model, dataset if dtype != "fp" else None, output,
                     platform_override=target, dtype_override=dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
