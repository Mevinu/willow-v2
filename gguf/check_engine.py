"""Compares the GGUF engine with the safetensors engine on the same model: tokenization, prompt logits and
greedy output. The engines are loaded one after the other, so only one model is in RAM at a time.

    .venv/bin/python -m gguf.check_engine ~/Models/Qwen3.6-35B-A3B ~/Models/Qwen3.6-35B-A3B-Q8_0.gguf

Both models need their expert packs (`main.py pack`). Expect cosine ~0.999+ for Q8_0 and somewhat lower for
4-bit quants; a mapping bug shows up as cosine far below 0.9 and a different greedy reply.
"""
import gc
import sys
import argparse

import torch
import torch.nn.functional as F

from safetensor.engine import load_engine as load_safetensors
from .engine import load_engine as load_gguf

def run(engine, prompt: str, max_new_tokens: int):
    ids = engine.prompt_ids(prompt)
    engine.reset()
    logits = engine.forward(ids).float()
    reply = engine.generate(prompt, max_new_tokens)[len(ids):]
    engine.experts.close()
    return ids, logits, reply, engine.tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Hugging Face model folder (safetensors)")
    parser.add_argument("gguf", help="The same model as a .gguf file")
    parser.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    results = []
    for name, load, path in [("safetensors", load_safetensors, args.model_dir), ("gguf", load_gguf, args.gguf)]:
        print(f"running {name}...", flush=True)
        results.append(run(load(path), args.prompt, args.max_new_tokens))
        gc.collect()
    (ids_a, logits_a, reply_a, tok), (ids_b, logits_b, reply_b, _) = results

    ok = ids_a == ids_b
    print(f"prompt tokens   {'match' if ok else 'DIFFER'} ({len(ids_a)} tokens)")
    if ok:
        cos = F.cosine_similarity(logits_a, logits_b, dim=-1)
        top1 = (logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean().item()
        print(f"prompt logits   cosine min {cos.min():.4f} mean {cos.mean():.4f}, top-1 agreement {top1 * 100:.0f}%")
        ok &= cos.min().item() > 0.9
    same = reply_a == reply_b
    print(f"greedy reply    {'match' if same else 'differs'} ({len(reply_a)} vs {len(reply_b)} tokens)")
    print(f"\nsafetensors: {tok.decode(reply_a, skip_special_tokens=True)}\ngguf:        {tok.decode(reply_b, skip_special_tokens=True)}")
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
