#!/usr/bin/env python3
"""Generate RKLLM input-embedding calibration samples for DeepSeek-OCR."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
VLM_DATA = ROOT / "examples" / "multimodal_model_demo" / "data"
VLM_DEMO = VLM_DATA.parent


class StopForward(Exception):
    """Stop after DeepSeek-OCR assembles the language-model embeddings."""


def cpu_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_payload(item) for item in value)
    if isinstance(value, list):
        return [cpu_payload(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="local DeepSeek-OCR model")
    parser.add_argument("--output", required=True, help="calibration manifest")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def reset_position_ids(model: torch.nn.Module) -> None:
    """Rebuild deterministic buffers omitted from the published checkpoint."""
    for module in model.modules():
        position_ids = getattr(module, "position_ids", None)
        num_positions = getattr(module, "num_positions", None)
        if isinstance(position_ids, torch.Tensor) and isinstance(num_positions, int):
            module.position_ids = torch.arange(
                num_positions, device=position_ids.device
            ).expand((1, -1))


def main() -> int:
    args = parse_args()
    if args.height != args.width or args.height % 64:
        raise SystemExit("DeepSeek-OCR requires a square size divisible by 64")
    if not torch.cuda.is_available():
        raise SystemExit("DeepSeek-OCR calibration requires CUDA")

    model_path = Path(args.path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    samples_dir = output.parent / f"{output.stem}_samples"
    output.parent.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    reset_position_ids(model)

    model_module = importlib.import_module(type(model).__module__)
    format_messages = model_module.format_messages
    text_encode = model_module.text_encode
    image_transform = model_module.BasicImageTransform(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
    )

    # Capture the exact boundary between the multimodal front end and the
    # DeepSeek language model, matching Rockchip's Qwen calibration helper.
    language_base = type(model.model).__mro__[1]
    original_forward = language_base.forward
    captured: list[dict[str, Any]] = []

    def capture_forward(
        _self: Any, *forward_args: Any, **forward_kwargs: Any
    ) -> Any:
        captured.append(forward_kwargs)
        raise StopForward

    language_base.forward = capture_forward
    manifest: list[dict[str, Any]] = []
    try:
        records = json.loads((VLM_DATA / "datasets.json").read_text(encoding="utf-8"))
        if args.limit is not None:
            records = records[:args.limit]
        for index, record in enumerate(tqdm(records, desc="DeepSeek-OCR calibration")):
            image_path = VLM_DEMO / record["image_path"] / record["image"]
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            image = ImageOps.pad(image, (args.width, args.height), color=(127, 127, 127))
            image_tensor = image_transform(image).to(torch.bfloat16)

            conversation = [
                {"role": "<|User|>", "content": f"<image>\n{record['input']}"},
                {"role": "<|Assistant|>", "content": ""},
            ]
            prompt = format_messages(
                conversations=conversation, sft_format="plain", system_prompt=""
            )
            text_before, text_after = prompt.split("<image>", maxsplit=1)
            token_ids = text_encode(tokenizer, text_before, bos=False, eos=False)
            image_mask = [False] * len(token_ids)

            queries = math.ceil((args.width // 16) / 4)
            image_tokens = ([128815] * queries + [128815]) * queries + [128815]
            token_ids.extend(image_tokens)
            image_mask.extend([True] * len(image_tokens))
            trailing_ids = text_encode(tokenizer, text_after, bos=False, eos=False)
            token_ids.extend(trailing_ids)
            image_mask.extend([False] * len(trailing_ids))
            token_ids.insert(0, 0)
            image_mask.insert(0, False)

            input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
            images_seq_mask = torch.tensor([image_mask], dtype=torch.bool, device="cuda")
            empty_crops = torch.zeros(
                (1, 3, args.height, args.width),
                dtype=torch.bfloat16,
                device="cuda",
            )
            global_image = image_tensor.unsqueeze(0).cuda()
            spatial_crop = torch.tensor([[1, 1]], dtype=torch.long)

            captured.clear()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(
                        input_ids=input_ids,
                        images=[(empty_crops, global_image)],
                        images_seq_mask=images_seq_mask,
                        images_spatial_crop=spatial_crop,
                    )
            except StopForward:
                pass
            if len(captured) != 1 or "inputs_embeds" not in captured[0]:
                raise RuntimeError("Failed to capture DeepSeek-OCR inputs_embeds")

            payload = cpu_payload(captured[0])
            sample_path = samples_dir / f"sample_{index}"
            with sample_path.open("wb") as stream:
                pickle.dump(payload, stream)
            manifest.append(
                {
                    "sample": str(sample_path),
                    "token_nums": int(payload["inputs_embeds"].shape[1]),
                }
            )
    finally:
        language_base.forward = original_forward

    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
