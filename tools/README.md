# One-command conversion (RKLLM v1.3.1)

`custom_export.py` wraps the RKLLM 1.3.1 APIs and upstream encoder examples.
It accepts any Hugging Face model family supported by the installed toolkit,
including the new Spark-X2.5 and LFM2.5-VL language components. Encoder support
is listed separately below: toolkit language-model support does not imply an
upstream vision/audio exporter exists.

## Setup

Conversion runs on **x86 Linux**, using Python **3.10–3.12**. Install the wheel
matching your interpreter; this example uses Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
export BUILD_CUDA_EXT=0  # required before installing on Python 3.12
python -m pip install --upgrade pip
python -m pip install ./rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp311-cp311-linux_x86_64.whl
python -m pip install -r rkllm-toolkit/packages/requirements.txt
```

Install `rknn-toolkit2` for encoder conversion. RWKV7 requires Python 3.12 and
`rkllm-toolkit/packages/requirements_rwkv7.txt`. Follow upstream's model-specific
requirements and access restrictions. The regular requirements pin
Transformers 5.8.0, Torch 2.6.0, and TorchVision 0.21.0; keep that Torch pair
matched if `torchvision::nms` fails to import.

## LLMs and platform/quantization selection

```bash
# One model, one target and dtype
python tools/custom_export.py -m /models/Qwen3-1.7B \
  -p RK3576 -q w4a16_g128 --dataset ./data_quant.json

# Common matrix across all four Rockchip targets (7 artifacts)
python tools/custom_export.py -m /models/Qwen3-1.7B -p ALL

# Every SDK-supported quantized dtype on each Rockchip target (19 artifacts)
python tools/custom_export.py -m /models/Qwen3-1.7B \
  -p ALL -q all --dataset ./data_quant.json

# Every supported dtype on just one target
python tools/custom_export.py -m /models/Qwen3-1.7B -p RK3562 -q all

# Floating point: no calibration generation or quantization
python tools/custom_export.py -m /models/Qwen3-1.7B -p RK3588 -q fp
```

The SDK 1.3.1 manual (section 3.1.3) specifies:

| Target | Quantized dtypes | NPU cores |
| --- | --- | --- |
| RK3576 | w4a16, w4a16_g32/g64/g128, w8a8 | 1–2 |
| RK3588 | w8a8, w8a8_g128/g256/g512 | 1–3 |
| RK3562 | w8a8, w4a16_g32/g64/g128, w4a8_g32 | 1 |
| RV1126B | w4a16, w4a16_g32/g64/g128, w8a8 | 1 |

Use the complete dtype spelling, for example `w4a16_g64`. `fp` disables
quantization. `-q all` enumerates the quantized entries only. Model architecture,
linear-layer dimensions, and available memory can further restrict successful
conversion; the table is the platform matrix, not a guarantee for every checkpoint.

`-p ALL` now includes RK3562 and RV1126B as well as RK3576 and RK3588. Without
`-q`, it builds W8A8 on all four and W4A16-g128 on RK3576, RK3562, and RV1126B.
A single target defaults to W4A16-g128 on RK3576/RV1126B and W8A8 on
RK3588/RK3562. Explicit unsupported combinations fail before loading a model.
For a dtype shared by all targets, such as W8A8, `-p ALL -q w8a8` produces four
artifacts. `-p ALL -q w4a16_g128` fails because RK3588 does not support it.

The toolkit API also exposes `-p CUDA -q w4a16_g32` for Jetson. CUDA is an
explicit language-only target, excluded from `ALL` and from VLM encoder export.

W4A16 defaults to `grq`, which requires a CUDA GPU. Other dtypes default to
`normal`. CPU conversion must select `--device cpu --quantized-algorithm normal`
for W4A16. Supported overrides are `none`, `normal`, `grq`, `normal_r1` through
`normal_r64`, and `grq_r1` through `grq_r64`. The rank variants export LoRA
correction weights and cannot be combined with `--model-lora`.

Other API options include `--device`, `--load-dtype`, `--model-lora`,
`--custom-config` (JSON file/object), `--optimization-level 0|1`,
`--num-npu-core`, `--hybrid-rate` (0 inclusive, 1 exclusive),
`--max-context` (multiple of 32, up to 16384), and `--extra-qparams`
(cached `.qparams` path or JSON). Use `--no-export-tokenizer` and/or
`--no-export-embedding` for the corresponding SDK export switches.

## GGUF

The SDK supports FP16 and Q4_0 GGUF inputs. Local `.gguf` files are detected
automatically; `--format gguf` selects the loader explicitly.

```bash
# Q4_0 corresponds to w4a16_g32; RK3588 cannot use that dtype.
python tools/custom_export.py -m /models/model-Q4_0.gguf \
  -p RK3576 -q w4a16_g32 --dataset ./data_quant.json

python tools/custom_export.py -m /models/model-F16.gguf -p RK3588 -q fp
```

Choose the dtype matching your input; the wrapper does not inspect GGUF tensor
types. Quantized GGUF conversion needs an explicit calibration manifest because
the automatic text calibration helper loads a Hugging Face model. GGUF LoRA
files use this same loader; Hugging Face `--model-lora` and `--custom-config`
options apply only to the Hugging Face loader.

## Vision and multimodal language components

```bash
python tools/custom_export.py --kind vlm -m /models/Qwen2.5-VL-3B-Instruct \
  -p RK3588 -q w8a8 --batch-size 1 --height 448 --width 448

# Gemma4: vision + audio + floating-point language model
python tools/custom_export.py --kind vlm -m /models/gemma-4-E2B-it \
  -p RK3588 -q fp --audio

# Gemma4 audio encoder only, with one RKNN per Rockchip target
python tools/custom_export.py --kind vlm -m /models/gemma-4-E2B-it \
  -p ALL --audio-only
```

Model names are inferred from paths; use `--model-name` for renamed checkpoints.
The available vision routes are:

| Model name | ONNX exporter |
| --- | --- |
| minicpm-v-2_6 | upstream export_vision.py |
| qwen2-vl | upstream export_vision_qwen2.py, steps 1 and 2 |
| qwen2_5-vl-3b | upstream export_vision.py |
| qwen3-vl | upstream export_vision.py |
| qwen3.5 | upstream export_vision.py |
| smolvlm | upstream export_vision.py |
| internvl3-1b | upstream export_vision.py |
| deepseekocr | upstream export_vision.py |
| gemma-4 | upstream export_vision_gemma4.py |

Gemma4 audio uses upstream `export_audio_gemma4.py`; its batch size, frame
count, and mel bins are fixed by upstream (1, 756, 128). Gemma4 vision also
requires batch size 1 and rounds image dimensions up to multiples of 48.
The legacy Qwen2 exporter runs on CPU. Other vision routes accept
`--vision-device cpu|cuda`, subject to the upstream exporter's device behavior.
Qwen2/2.5 dimensions must be multiples of 28; Qwen3/3.5 dimensions must be
multiples of 32.

RKNN export uses `export_encoder_rknn.py`, mirroring the SDK 1.3.1 input shapes
and normalization, checking return codes, and releasing the RKNN instance.
Normalization follows the chosen model family, even for renamed checkpoints.
Encoders are exported without quantization, as in the upstream examples.

`--vision-only` omits the language component; `--skip-vision` omits the vision
component. `--audio` adds audio; `--audio-only` omits vision and language.
Use `--vision-stage onnx|rknn|all` and `--audio-stage onnx|rknn|all` to select
encoder stages. ONNX is shared across target platforms and exported once.
For RKNN-only conversion, put the ONNX (and any external tensor files) at
`output/<model>/<model>_vision.onnx` or `<model>_audio.onnx`, or under the root
chosen with `--output-dir`.

The SDK lists additional language-supported families (including LFM2.5-VL,
Gemma3/3n, Janus, and InternVL2) without providing a matching encoder script in
this release. Export their language component with `--kind vlm --skip-vision`
and an explicit calibration dataset (or `-q fp`). Full encoder conversion for
these families requires a separate model-specific exporter; choosing an
unavailable vision route fails explicitly.

## Calibration and output

Text LLMs generate a calibration dataset with upstream `generate_data_quant.py`
when `--dataset` is omitted. The VLM calibration helper supports Qwen2-VL,
Qwen2.5-VL, Qwen3-VL, and Qwen3.5 only, using the upstream demo images and prompts.
`--model-type` selects one of those helpers explicitly. Other VLMs require a
prepared multimodal manifest for quantization, such as Gemma4 with
`--dataset /data/gemma4_inputs.json`. A missing or invalid explicit manifest
fails instead of being silently regenerated. Floating-point conversion skips
calibration entirely unless `--prepare-dataset` is requested.

Calibration JSON uses the SDK text format `[{"input":"...","target":"..."}]`
or its multimodal `sample`/`token_nums` format. Relative multimodal sample paths
resolve against the manifest's directory. Automatically generated multimodal
samples are copied into the model's output tree so later conversions cannot
overwrite their backing files.

By default all outputs live under `output/<model>/`:

- `<model>_data_quant.json` and optional `calibration/` samples
- `<model>_vision.onnx` / `<model>_audio.onnx` (including external tensor files)
- `<platform>/<model>_<platform>_<dtype>.rkllm`
- `<platform>/<model>_vision_<platform>.rknn` / `<model>_audio_<platform>.rknn`

Local paths below this repository's `models/` preserve nested directories in
`output/`. `--output-dir` overrides the common root. `-o` overrides the language
artifact path; multiple builds fan out below its parent into platform directories
and append platform/dtype to the requested stem. Existing artifacts are skipped;
`--force` rebuilds them. Use a distinct output root or `--force` when changing
checkpoint, shapes, or conversion settings. Upstream exporters use shared
working files: run one conversion command at a time per checkout.

## Validation and SDK updates

```bash
python tools/custom_export.py --help
python -m unittest discover -s tests
```

The tests use fake toolkit APIs to cover the 19-build matrix, API dispatch,
resume behavior, calibration safety, encoder routing, external ONNX tensors,
and error propagation. Real model conversion and on-device inference still
need validation on the supported Linux/toolkit/hardware environment.

Keep upstream examples unchanged during SDK updates. Compare the new wheel's
`rkllm/api/rkllm.py`, SDK manual, and encoder scripts; update the compatibility
layer, matrix, tests, and this guide together. The current upstream baseline is
[`release-v1.3.1`](https://github.com/airockchip/rknn-llm/releases/tag/release-v1.3.1).
