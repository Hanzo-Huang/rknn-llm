# Convenient model conversion

`custom_export.py` is a repository-owned compatibility wrapper around the
Rockchip-generated conversion examples. It does not modify those examples, so
updating the SDK remains a reviewable operation.

The wrapper uses `run_vision.py` for both VLM vision stages. Use
`--vision-stage onnx` to generate only ONNX, `--vision-stage rknn` to convert
an existing ONNX to RKNN, or the default `--vision-stage all` for both. This
keeps Qwen2.5-VL and RKNN/ONNX compatibility fixes outside Rockchip's generated
exporters.

## Setup

Use a supported Python version (3.9–3.12) and install the toolkit matching the
SDK checkout. For this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
export BUILD_CUDA_EXT=0       # required before install on Python 3.12
python -m pip install --upgrade pip
python -m pip install ./rkllm-toolkit/packages/rkllm_toolkit-1.3.0-cp311-cp311-linux_x86_64.whl
python -m pip install -r rkllm-toolkit/packages/requirements.txt
```

Install `rknn-toolkit2` as well when converting the VLM vision component.
Choose the wheel whose CPython tag matches the active interpreter. Conversion
is normally performed on an x86 Linux host with a CUDA GPU for LLM loading.

### Torch/TorchVision troubleshooting

If VLM export fails with `RuntimeError: operator torchvision::nms does not
exist`, Torch and TorchVision are mismatched or one package was installed
without its compiled operators. Reinstall the pinned pair in the active
environment:

```bash
python -m pip uninstall -y torch torchvision
python -m pip install torch==2.6.0 torchvision==0.21.0
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

For a CUDA-specific installation, use the matching PyTorch index for the CUDA
version on the host. Do not mix a CPU Torch package with a CUDA TorchVision
package, or packages from different PyTorch release lines.

## LLM conversion

The wrapper accepts model path, platform, quantization, dataset, output, and
all commonly changed RKLLM build values. Omitted options use the defaults
shown by `python tools/custom_export.py --help`.

```bash
python tools/custom_export.py -m /models/Qwen2.5-1.5B-Instruct \
  -p RK3588 -q w8a8 -c 4096 \
  --dataset ./data_quant.json
```

For the usual multi-platform conversion, omit `-q`:

```bash
python tools/custom_export.py \
  -m /models/MyModel \
  -p all
```

This generates one calibration dataset, then builds `w4a16_g128` and `w8a8`
for RK3576 and `w8a8` for RK3588.

The short options `-m`, `-p`, `-q`, `-O`, `-o`, and `-c` are provided for
compatibility with the former shell entry point. With `-p ALL`, `-o` is used
as a base path and outputs are placed below per-platform subdirectories.

Without `--dataset`, the default is
`output/<model>/<model>_data_quant.json`. For a quantized LLM this file is
generated automatically with the repository's calibration helper.

With `--platform ALL` and no explicit `--dtype`, the wrapper builds exactly
three artifacts: `w4a16_g128` and `w8a8` for RK3576, plus `w8a8` for RK3588.

For floating-point output use `--dtype fp`; this sets `do_quantization=False`.
For group quantization use `w4a16_g32`, `w4a16_g64`, or `w4a16_g128`. The
quantization algorithm defaults to `normal` for W8 and `grq` for W4, and can
be overridden with `--quantized-algorithm`. The toolkit also accepts
`w8a8_g128`, `w8a8_g256`, and `w8a8_g512`.

`--max-context` must be a multiple of 32 and no larger than 16384, matching
the RKLLM 1.3.0 API requirement.

Useful overrides include `--device`, `--load-dtype`, `--optimization-level`,
`--max-context`, `--num-npu-core`, `--hybrid-rate`, `--extra-qparams`,
`--model-lora`, and `--custom-config`.

## VLM conversion

VLM conversion runs the existing vision exporter and RKNN exporter, then uses
the same wrapper for the language RKLLM component. It writes artifacts below
`output/<model>/<platform>/` and the calibration file below `output/<model>/`
by default. VLM vision export uses `cuda` by default; use `--vision-device cpu`
only when GPU export is unavailable.

```bash
python tools/custom_export.py --kind vlm \
  --model /models/Qwen2.5-VL-3B-Instruct \
  --platform RK3588 --dtype w8a8 \
  --batch-size 1 --height 448 --width 448
```

You do not normally need to enter the VLM model type. The wrapper infers the
vision name from common model path names (`qwen3-vl`, `qwen2.5-vl`, `qwen3.5`,
`MiniCPM-V-2_6`, `SmolVLM`, `InternVL3`, and `DeepSeekOCR`) and infers the
calibration type for Qwen VLMs. Pass `--model-name` or `--model-type` explicitly
only when the path is ambiguous.

Generate only one vision artifact when needed:

```bash
# Hugging Face/PyTorch model -> ONNX
python tools/custom_export.py --kind vlm --model /models/Qwen2.5-VL-3B-Instruct \
  --vision-stage onnx --vision-only

# Existing ONNX -> RKNN (the ONNX must be in the upstream export/onnx directory)
python tools/custom_export.py --kind vlm --model /models/Qwen2.5-VL-3B-Instruct \
  --vision-stage rknn --vision-only
```

For supported Qwen VLMs, calibration data is generated automatically when it
does not exist. It can also be generated explicitly:

```bash
python tools/custom_export.py --kind vlm --model /models/Qwen3-VL \
  --prepare-dataset \
  --platform RK3588 --dtype w8a8
```

Use `--skip-vision` when only the VLM language component is needed. The
vision exporter currently supports `minicpm-v-2_6`, `qwen2_5-vl-3b`, `qwen3-vl`,
`qwen3.5`, `smolvlm`, `internvl3-1b`, and `deepseekocr`.

## Updating for a new Rockchip SDK

When Rockchip releases a new version (for example v1.3.1):

1. Replace or add the toolkit wheel and update the version in this document.
2. Diff the new generated `export_rkllm.py`, VLM `export_rkllm.py`, and vision
   exporters against the current files. Do not edit the generated files to
   implement local convenience behavior.
3. Update `tools/custom_export.py` only where an API argument, dtype, platform,
   model wrapper, or generated output path changed. Keep the upstream path
   constants and the mapping in this section together.
4. Run `python tools/custom_export.py --help`, then perform one small `fp`
   conversion and one quantized conversion for each affected model family.
5. Record the SDK version and any compatibility change in `CHANGELOG.md`.

The wrapper intentionally passes through explicit values instead of hiding
them in a project config. That makes SDK diffs easy to audit and future AI or
human developers can update one compatibility layer when the upstream examples
change.
