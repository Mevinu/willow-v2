"""Loads a Qwen3-MoE or Qwen3.5/3.6-MoE GGUF (as made by llama.cpp's convert_hf_to_gguf.py) into model.py's
engines. The non-expert weights are dequantized to bf16 at load (bf16 matmuls run faster than torch's int4 kernel
for them on Apple Silicon) and renamed to their Hugging Face names, undoing the changes llama.cpp's converter
makes. The routed experts stay quantized in the pack (see gguf.store)."""
import os
import re

import torch

from gguf.compiler import EXPERT_RE, default_pack_dir, num_decoder_layers
from gguf.quants import QuantizedEmbedding, dequantize
from gguf.store import ExpertStore
from gguf.tokenizer import build_tokenizer
from .model import MoEEngine, Qwen35MoEEngine

ARCHITECTURES = {"qwen3moe": MoEEngine, "qwen35moe": Qwen35MoEEngine}  # GGUF general.architecture -> engine

# llama.cpp name -> Hugging Face name, relative to the layer (blk.N. -> <prefix>layers.N.)
LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",             # Qwen3-MoE
    "post_attention_norm.weight": "post_attention_layernorm.weight",  # Qwen3.5-MoE
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}
GLOBAL_NAMES = {"token_embd.weight": "embed_tokens.weight", "output_norm.weight": "norm.weight"}
LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
KEEP_FLOAT32 = (".linear_attn.A_log", ".linear_attn.dt_bias")  # the engine computes with these in float32
# Zero-centered norms whose names don't end in "norm.weight" (Qwen3.8's PLE), which the converter also stored as 1 + w
OTHER_ZERO_CENTERED = (".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight")
FIXED_ARCHITECTURES = ("qwen35moe", "qwen4exp")  # whose tensors go through fix_qwen35_tensor
END_TOKENS = ("<|im_end|>", "<|endoftext|>")  # Qwen's generation_config.json stops on both

def build_config(metadata: dict, tensors: dict) -> dict:
    """Rebuilds the Hugging Face config.json fields the engine reads from the GGUF metadata."""
    arch = metadata["general.architecture"]

    def get(key, default=None):
        return metadata.get(f"{arch}.{key}", default)

    heads, hidden = get("attention.head_count"), get("embedding_length")
    head_dim = get("attention.key_length") or hidden // heads
    num_layers = num_decoder_layers(metadata)
    cfg = {
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": get("attention.head_count_kv"),
        "head_dim": head_dim,
        "num_experts": get("expert_count"),
        "num_experts_per_tok": get("expert_used_count"),
        "norm_topk_prob": get("expert_weights_norm", True),
        "rms_norm_eps": get("attention.layer_norm_rms_epsilon"),
        "max_position_embeddings": get("context_length"),
    }

    # q_proj emits [query | gate] per head when the attention output is gated (Qwen3.5)
    full_layers = [l for l in range(num_layers) if f"blk.{l}.attn_q.weight" in tensors]
    if full_layers:
        cfg["attn_output_gate"] = tensors[f"blk.{full_layers[0]}.attn_q.weight"].shape[0] == 2 * heads * head_dim

    rope = {"rope_theta": get("rope.freq_base", 10000.0), "rope_type": "default",
            "partial_rotary_factor": get("rope.dimension_count", head_dim) / head_dim}
    if get("rope.scaling.type") not in (None, "none"):
        rope.update(rope_type=get("rope.scaling.type"), factor=get("rope.scaling.factor"),
                    original_max_position_embeddings=get("rope.scaling.original_context_length"))
        for gguf_key, hf_key in [("yarn_beta_fast", "beta_fast"), ("yarn_beta_slow", "beta_slow"),
                                 ("yarn_attn_factor", "attention_factor"), ("yarn_truncate", "truncate")]:
            if get(f"rope.scaling.{gguf_key}") is not None:
                rope[hf_key] = get(f"rope.scaling.{gguf_key}")
    cfg["rope_parameters"] = rope

    if get("ssm.conv_kernel") is not None:  # Gated DeltaNet layers (Qwen3.5/3.6/3.8)
        recurrent = get("attention.recurrent_layers")
        if recurrent is None:
            interval = get("full_attention_interval", 4)
            recurrent = [(l + 1) % interval != 0 for l in range(num_layers)]
        cfg["layer_types"] = ["linear_attention" if r else "full_attention" for r in recurrent[:num_layers]]
        v_heads = get("ssm.time_step_rank")
        cfg.update(linear_num_key_heads=get("ssm.group_count"), linear_num_value_heads=v_heads,
                   linear_key_head_dim=get("ssm.state_size"), linear_value_head_dim=get("ssm.inner_size") // v_heads,
                   linear_conv_kernel_dim=get("ssm.conv_kernel"))
    return cfg

def v_heads_to_grouped(t: torch.Tensor, dim: int, num_k_heads: int, v_per_k: int, head_dim: int) -> torch.Tensor:
    """Undoes llama.cpp's linear-attention V-head reorder. When there are more V heads than K heads, the converter
    stores them tiled ([K0v0, K1v0, ..., K0v1, K1v1, ...]) for ggml's broadcasting; Hugging Face and this engine
    group them by K head ([K0v0, K0v1, ..., K1v0, K1v1, ...])."""
    shape = list(t.shape)
    t = t.reshape(shape[:dim] + [v_per_k, num_k_heads, head_dim] + shape[dim + 1:])
    return t.transpose(dim, dim + 1).reshape(shape)

def fix_qwen35_tensor(name: str, t: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Reverses what llama.cpp's converter does to Qwen3.5/3.6/3.8 tensors (Qwen3NextModel and
    _LinearAttentionVReorderBase in conversion/qwen.py, and Qwen4ExpTextModel in conversion/qwen4exp.py). t is
    float32."""
    if (name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight")) or name.endswith(OTHER_ZERO_CENTERED):
        t = t - 1  # the converter adds 1 to zero-centered RMSNorm weights; the engine adds it back
    elif name.endswith("linear_attn.A_log"):
        t = torch.log(-t)  # stored as A = -exp(A_log)
    elif name.endswith("linear_attn.conv1d.weight"):
        t = t.unsqueeze(1)  # [channels, kernel] -> [channels, 1, kernel]
    elif name.endswith("mlp.shared_expert_gate.weight") and t.dim() == 1:
        t = t.unsqueeze(0)

    k_heads, v_heads = cfg.get("linear_num_key_heads", 0), cfg.get("linear_num_value_heads", 0)
    if ".linear_attn." not in name or k_heads == v_heads:
        return t
    v_per_k, k_dim, v_dim = v_heads // k_heads, cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    qk_rows = 2 * k_heads * k_dim
    if name.endswith((".in_proj_qkv.weight", ".conv1d.weight")):
        return torch.cat((t[:qk_rows], v_heads_to_grouped(t[qk_rows:], 0, k_heads, v_per_k, v_dim)))
    if name.endswith(".in_proj_z.weight"):
        return v_heads_to_grouped(t, 0, k_heads, v_per_k, v_dim)
    if name.endswith((".in_proj_a.weight", ".in_proj_b.weight", ".A_log", ".dt_bias")):
        return v_heads_to_grouped(t.unsqueeze(-1), 0, k_heads, v_per_k, 1).reshape(t.shape)
    if name.endswith(".out_proj.weight"):
        return v_heads_to_grouped(t, 1, k_heads, v_per_k, v_dim)
    return t

def hf_names(gguf_name: str, prefix: str, cfg: dict, tensors: dict, layer_names: dict = None,
             global_names: dict = None):
    """Maps one llama.cpp tensor to [(Hugging Face name, row slice or None)]; [] for tensors the engine skips.
    layer_names and global_names default to LAYER_NAMES and GLOBAL_NAMES; a global name mapped to None is skipped."""
    layer_names = LAYER_NAMES if layer_names is None else layer_names
    global_names = GLOBAL_NAMES if global_names is None else global_names
    if gguf_name == "output.weight":
        return [("lm_head.weight", None)]
    if gguf_name in global_names:
        return [] if global_names[gguf_name] is None else [(prefix + global_names[gguf_name], None)]
    m = LAYER_RE.match(gguf_name)
    if not m:
        raise NotImplementedError(f"Don't know where GGUF tensor {gguf_name} goes")
    layer, rest = int(m.group(1)), m.group(2)
    if layer >= cfg["num_hidden_layers"] or EXPERT_RE.match(gguf_name):
        return []  # multi-token-prediction layers, and the routed experts (read from the pack instead)
    p = f"{prefix}layers.{layer}."
    if rest == "attn_qkv.weight":
        if cfg.get("layer_types", ["full_attention"])[layer] == "linear_attention":
            return [(p + "linear_attn.in_proj_qkv.weight", None)]
        # Fused full-attention q/k/v (convert_hf_to_gguf.py --fuse-qkv): split the rows back up
        kv_rows = cfg["num_key_value_heads"] * cfg["head_dim"]
        q_rows = tensors[gguf_name].shape[0] - 2 * kv_rows
        return [(p + "self_attn.q_proj.weight", slice(0, q_rows)),
                (p + "self_attn.k_proj.weight", slice(q_rows, q_rows + kv_rows)),
                (p + "self_attn.v_proj.weight", slice(q_rows + kv_rows, None))]
    if rest.endswith(".bias") and rest not in layer_names:
        # Biases sit next to their weight: blk.N.attn_q.bias -> ...q_proj.bias
        weight_name = layer_names.get(rest.removesuffix(".bias") + ".weight")
        if weight_name:
            return [(p + weight_name.removesuffix(".weight") + ".bias", None)]
    if rest not in layer_names:
        raise NotImplementedError(f"Don't know where GGUF tensor {gguf_name} goes")
    return [(p + layer_names[rest], None)]

def load_weights(arch: str, prefix: str, cfg: dict, tensors: dict, layer_names: dict = None,
                 global_names: dict = None) -> dict:
    """Reads and dequantizes every non-expert tensor (named by hf_names). Returns {Hugging Face name: tensor}, all
    resident in RAM. A quantized embedding table stays quantized (QuantizedEmbedding), unless it doubles as the
    output layer."""
    weights = {}
    embed_name = prefix + GLOBAL_NAMES["token_embd.weight"]
    separate_output = "output.weight" in tensors
    fds = {}
    try:
        for gguf_name, t in tensors.items():
            targets = hf_names(gguf_name, prefix, cfg, tensors, layer_names, global_names)
            if not targets:
                continue
            if t.path not in fds:
                fds[t.path] = os.open(t.path, os.O_RDONLY)
            raw = bytearray(t.nbytes)
            if os.preadv(fds[t.path], [raw], t.offset) != t.nbytes:
                raise IOError(f"short read for {gguf_name} in {t.path}")
            for name, rows in targets:
                if name == embed_name and separate_output and QuantizedEmbedding.supports(t.type, t.shape):
                    weights[name] = QuantizedEmbedding(raw, t.type, t.shape)
                    continue
                # Tensors that need the converter's changes undone go through float32; the rest (including the
                # big embedding and output tables) dequantize straight to bf16
                fix = arch in FIXED_ARCHITECTURES and (".linear_attn." in name or name.endswith(
                    ("norm.weight", "_gate.weight") + OTHER_ZERO_CENTERED))
                keep_float32 = name.endswith(KEEP_FLOAT32)
                tensor = dequantize(raw, t.type, t.shape, torch.float32 if fix or keep_float32 else torch.bfloat16)
                if rows is not None:
                    tensor = tensor[rows]
                if fix:
                    tensor = fix_qwen35_tensor(name, tensor, cfg)
                weights[name] = tensor.to(torch.float32 if keep_float32 else torch.bfloat16).contiguous()
    finally:
        for fd in fds.values():
            os.close(fd)
    return weights

def load_engine(gguf_path: str, metadata: dict, tensors: dict, pack_dir: str = None, cache_gb: float = None,
                preload: bool = True, gpu_backbone: bool = True, sync_timings: bool = False,
                min_expert_weight: float = 0.0) -> MoEEngine:
    """metadata and tensors are the GGUF's header (gguf.reader.read_gguf). cache_gb None or 0 (no --cache-gb) turns
    the expert cache off. preload reads the most used experts into the cache first (see ExpertStore.preload). With
    a GPU pack, gpu_backbone also runs everything else (attention, the router, the shared expert, the output layer)
    on the GPU. sync_timings makes the per-section decode times real when the GPU is used, at some cost in speed
    (see MoEEngine.tick). min_expert_weight skips routed experts below that weight (see
    MoEEngine.skip_small_experts)."""
    arch = metadata["general.architecture"]
    engine_class = ARCHITECTURES[arch]
    cfg = build_config(metadata, tensors)
    weights = load_weights(arch, engine_class.prefix, cfg, tensors)
    return build_engine(engine_class, gguf_path, metadata, cfg, weights, pack_dir, cache_gb, preload, gpu_backbone,
                        sync_timings, min_expert_weight)

def build_engine(engine_class, gguf_path: str, metadata: dict, cfg: dict, weights: dict, pack_dir: str,
                 cache_gb: float, preload: bool, gpu_backbone: bool, sync_timings: bool, min_expert_weight: float,
                 **engine_options):
    """The part of loading that every Qwen GGUF shares: the tokenizer and end tokens, the expert store, the device,
    and the engine itself (engine_options are passed on to its constructor)."""
    tokenizer = build_tokenizer(metadata)
    vocab = tokenizer.get_vocab()
    eos_ids = {metadata[k] for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id") if k in metadata}
    eos_ids |= {vocab[t] for t in END_TOKENS if t in vocab}
    cfg["bos_token_id"] = metadata.get("tokenizer.ggml.bos_token_id", vocab.get("<|endoftext|>"))

    experts = ExpertStore(pack_dir or default_pack_dir(gguf_path), int((cache_gb or 0) * 1024**3))
    if preload:
        experts.preload()
    device = "mps" if experts.gpu and gpu_backbone else "cpu"
    engine = engine_class(cfg, eos_ids, weights, experts, tokenizer, device, **engine_options)
    engine.sync_timings = sync_timings
    engine.min_expert_weight = min_expert_weight
    print(f"experts on the {'GPU' if experts.gpu else 'CPU'}, the rest on the {'GPU' if device == 'mps' else 'CPU'}")
    return engine
