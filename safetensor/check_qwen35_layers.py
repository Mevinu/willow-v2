"""Compares safetensor/engine.py's Qwen3.5 decoder layers against transformers' own layer classes, using the real
Qwen3.6-35B-A3B weights one layer at a time (the whole model does not fit in 16 GB).
Needs transformers 5.x, so run it with .venv-ref (see requirements-ref.txt):

    .venv-ref/bin/python -m safetensor.check_qwen35_layers /path/to/Qwen3.6-35B-A3B 0 3 39

Expect cosine ~0.9995+ in bf16; a real bug shows up as cosine far below 0.99.
"""
import sys, time
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding,
)

from .compiler import build_index_map
from .engine import load_engine

model_dir = sys.argv[1]
layers = [int(x) for x in sys.argv[2:]] or [0, 3]

t0 = time.time()
engine = load_engine(model_dir)
print(f"engine loaded in {time.time() - t0:.0f}s")
config = AutoConfig.from_pretrained(model_dir).text_config
config._attn_implementation = "sdpa"
index_map = build_index_map(model_dir)

# Realistic input: embeddings of a real prompt (24 tokens)
ids = engine.prompt_ids("Explain in two sentences why the sky is blue, and mention Rayleigh scattering.")[:24]
x = engine.w(engine.prefix + "embed_tokens.weight")[torch.tensor(ids)]
T, split = x.shape[0], 20

rotary = Qwen3_5MoeTextRotaryEmbedding(config)
position_ids = torch.arange(T)[None]
cos_sin = rotary(x[None], position_ids)

def engine_layer(inp, layer):
    p = f"{engine.prefix}layers.{layer}"
    cos, sin = engine.rope(inp.shape[0])
    h = inp + engine.token_mixer(engine.rms_norm(inp, p + ".input_layernorm.weight"), layer, cos, sin)
    out = h + engine.moe(engine.rms_norm(h, p + ".post_attention_layernorm.weight"), layer)
    engine.position += inp.shape[0]
    return out

for layer in layers:
    hf = Qwen3_5MoeDecoderLayer(config, layer).to(torch.bfloat16).eval()
    prefix = f"{engine.prefix}layers.{layer}."
    state = {}
    for name in index_map:
        if name.startswith(prefix):
            with safe_open(index_map[name], "pt") as f:
                state[name[len(prefix):]] = f.get_tensor(name)
    missing, unexpected = hf.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    with torch.no_grad():
        theirs = hf(x[None], position_embeddings=cos_sin, position_ids=position_ids)[0]
        theirs = theirs[0] if theirs.dim() == 3 else theirs

        engine.reset()
        ours = torch.cat([engine_layer(x[:split], layer)] + [engine_layer(x[t:t + 1], layer) for t in range(split, T)])

    delta_in, delta_ours, delta_theirs = x.float(), (ours - x).float(), (theirs - x).float()
    rel = (delta_ours - delta_theirs).norm() / delta_theirs.norm()
    cos_sim = torch.nn.functional.cosine_similarity(delta_ours.flatten(), delta_theirs.flatten(), dim=0)
    print(f"layer {layer:2d} ({engine.layer_types[layer]:16s}) layer-output rel err = {rel:.2e}, cosine = {cos_sim:.6f}  "
          f"[prefill {split} + {T - split} cached steps]")
    del hf, state
