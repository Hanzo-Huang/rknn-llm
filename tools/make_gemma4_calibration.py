#!/usr/bin/env python3
"""Capture Gemma4 multimodal calibration using the SDK's image/prompt dataset.

Use fixed square preprocessing and token counts matching export_vision_gemma4.py,
rather than the HF processor's variable-resolution image budget.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.height % 48 or args.width % 48:
        parser.error('Gemma4 dimensions must be multiples of 48')

    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    source = ROOT / 'examples/multimodal_model_demo/export/modeling_gemma4.py'
    spec = importlib.util.spec_from_file_location('transformers.models.gemma4.modeling_gemma4', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import transformers.models.gemma4 as pkg
    for name in module.__all__:
        setattr(pkg, name, getattr(module, name))
    import transformers.models.auto as auto
    for name in dir(auto):
        mapping = getattr(auto, name, None)
        if isinstance(getattr(mapping, '_modules', None), dict):
            mapping._modules.pop('gemma4', None)

    model = AutoModelForCausalLM.from_pretrained(args.model, config=AutoConfig.from_pretrained(args.model),
              dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager', local_files_only=True).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    data_root = ROOT / 'examples/multimodal_model_demo'
    records = json.loads((data_root / 'data/datasets.json').read_text())
    if args.limit:
        records = records[:args.limit]
    output = args.output.resolve()
    samples = output.parent / 'gemma4_samples'
    samples.mkdir(parents=True, exist_ok=True)
    captured = []

    class Captured(Exception): pass
    def cpu(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return type(value)(cpu(v) for v in value)
        return value
    def hook(module, positional, kwargs):
        captured.append(cpu(kwargs))
        raise Captured()
    handle = model.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
    image_tokens = args.height // 48 * (args.width // 48)
    rows=[]
    try:
        for index, record in enumerate(records):
            image = Image.open(data_root / record['image_path'] / record['image']).convert('RGB')
            size=max(image.size)
            square=Image.new('RGB',(size,size),(127,127,127))
            square.paste(image,((size-image.width)//2,(size-image.height)//2))
            square=square.resize((args.width,args.height),Image.Resampling.BILINEAR)
            pixels=torch.from_numpy(np.asarray(square,dtype=np.float32).copy()/255)
            patches=pixels.reshape(args.height//16,16,args.width//16,16,3).permute(0,2,1,3,4).reshape(1,-1,16*16*3)
            y,x=torch.meshgrid(torch.arange(args.height//16),torch.arange(args.width//16),indexing='ij')
            positions=torch.stack([x,y],dim=-1).reshape(1,-1,2)
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':record['input']}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            text=text.replace(processor.image_token, processor.boi_token+processor.image_token*image_tokens+processor.eoi_token)
            inputs=processor(text=[text],return_tensors='pt',return_mm_token_type_ids=True)
            actual=int((inputs['input_ids']==model.config.image_token_id).sum())
            if actual != image_tokens: raise RuntimeError(f'Expected {image_tokens} image tokens, got {actual}')
            inputs['pixel_values']=patches
            inputs['image_position_ids']=positions
            inputs={k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            try:
                with torch.inference_mode(): model(**inputs,use_cache=False)
            except Captured: pass
            if len(captured)!=1: raise RuntimeError('Language inputs were not captured exactly once')
            value=captured.pop()
            path=samples/f'sample_{index:03d}.pkl'
            with path.open('wb') as file: pickle.dump(value,file)
            rows.append({'sample':str(path),'token_nums':int(value['inputs_embeds'].shape[1])})
            print(json.dumps({'sample':index,'tokens':rows[-1]['token_nums'],'image_tokens':image_tokens,'keys':list(value)}),flush=True)
    finally:
        handle.remove()
    output.write_text(json.dumps(rows,indent=2)+'\n')
    print(f'Wrote {len(rows)} samples to {output}',flush=True)


if __name__=='__main__':main()
