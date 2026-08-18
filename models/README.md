# Models

Place downloaded Hugging Face model directories here. Model weights are ignored
by Git.

Example:

```text
models/
└── deepseek-ai/
    └── DeepSeek-OCR/
```

Convert a local model with:

```bash
python tools/custom_export.py \
  --model models/deepseek-ai/DeepSeek-OCR \
  --kind vlm \
  --platform all
```
