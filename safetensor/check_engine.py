"""Checks safetensor/engine.py against Hugging Face transformers on the same model: logits and greedy output must match."""
import sys
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .engine import load_engine, rope_parameters

def load_reference(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir)
    # Configs saved by transformers 5.x use field names that 4.x silently ignores, so map them back
    rope = rope_parameters(config.to_dict())
    config.rope_theta = rope.pop("rope_theta")
    config.rope_scaling = rope if rope["rope_type"] != "default" else None
    if hasattr(config, "num_local_experts"):
        config.num_experts = config.num_local_experts

    model, info = AutoModelForCausalLM.from_pretrained(model_dir, config=config, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if k != "lm_head.weight"]
    if missing or info["unexpected_keys"] or info["mismatched_keys"]:
        sys.exit(f"Reference model did not load cleanly: {info}")
    return model.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    engine = load_engine(args.model_dir)
    reference = load_reference(args.model_dir)
    ok = True

    # 1. Prompt logits in one pass (prefill)
    ids = [engine.cfg["bos_token_id"]] + engine.tokenizer.encode(args.prompt, add_special_tokens=False).ids
    engine.reset()
    ours = engine.forward(ids)
    with torch.no_grad():
        theirs = reference(torch.tensor([ids])).logits[0]
    diff = (ours - theirs).abs().max().item()
    print(f"prefill logits   max abs diff = {diff:.2e}  ({len(ids)} tokens)")
    ok &= diff < args.atol

    # 2. Same logits, one token at a time (exercises the KV cache and RoPE positions)
    engine.reset()
    stepwise = torch.cat([engine.forward([t]) for t in ids])
    diff = (stepwise - theirs).abs().max().item()
    print(f"stepwise logits  max abs diff = {diff:.2e}")
    ok &= diff < args.atol

    # 3. Greedy generation
    ours_ids = engine.generate(args.prompt, args.max_new_tokens)
    with torch.no_grad():
        theirs_ids = reference.generate(
            torch.tensor([ids]), max_new_tokens=args.max_new_tokens, do_sample=False,
            eos_token_id=engine.cfg["eos_token_id"], pad_token_id=engine.cfg.get("pad_token_id"),
        )[0].tolist()
    if theirs_ids and theirs_ids[-1] == engine.cfg["eos_token_id"]:
        theirs_ids = theirs_ids[:-1]
    match = ours_ids == theirs_ids
    print(f"greedy tokens    {'match' if match else 'DIFFER'} ({len(ours_ids) - len(ids)} generated)")
    ok &= match

    print("\n" + engine.tokenizer.decode(ours_ids, skip_special_tokens=True))
    if not match:
        print("\nreference:\n" + engine.tokenizer.decode(theirs_ids, skip_special_tokens=True))
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
