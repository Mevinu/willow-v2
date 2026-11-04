"""Runs GGUF models (as made by llama.cpp's convert_hf_to_gguf.py) on the same model code as the safetensors
engine. The non-expert weights are dequantized to bf16 at load and renamed to their Hugging Face names, undoing
the changes llama.cpp's converter makes; the routed experts stay quantized in the pack. Q4_K experts are
multiplied 4-bit with torch's int4 kernel, and other types are dequantized as each one is read."""
import os
import re

import torch
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from safetensor.engine import ExpertStore, MoEEngine, Qwen35MoEEngine
from .compiler import EXPERT_RE, default_pack_dir, num_decoder_layers
from .quants import PACK_LAYOUTS, Int4Weight, dequant_int8_g16, dequantize
from .reader import read_gguf

ENGINES = {"qwen3moe": MoEEngine, "qwen35moe": Qwen35MoEEngine}

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

# Pre-tokenizer regexes, from the models' tokenizer.json (llama.cpp's tokenizer.ggml.pre names them)
PRE_TOKENIZER_REGEXES = {
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen35": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
CONTROL_TOKEN, USER_DEFINED_TOKEN = 3, 4  # tokenizer.ggml.token_type values
END_TOKENS = ("<|im_end|>", "<|endoftext|>")  # Qwen's generation_config.json stops on both

class GGUFExpertStore(ExpertStore):
    """Reads experts from a pack built from a GGUF file. INT4_G32 parts stay 4-bit (Int4Weight); the rest are
    dequantized to bf16 on the read threads."""

    pack_format = "gguf"

    def __init__(self, pack_dir: str, cache_bytes: int = 0):
        super().__init__(pack_dir, cache_bytes)
        types = {type_name for _, type_name in self.parts.values()}
        if "INT4_G32" in types and self.manifest.get("torch_version") != torch.__version__:
            raise ValueError(f"{pack_dir} was built with torch {self.manifest.get('torch_version')}, and its int4 "
                             f"layout may not match torch {torch.__version__}; rerun `main.py pack` on this model")
        if types & set(PACK_LAYOUTS):
            print(f"note: {pack_dir} stores {', '.join(sorted(types & set(PACK_LAYOUTS)))} experts as raw ggml "
                  "blocks, which are slow to dequantize; rerun `main.py pack` on this model to convert them")

    def remember(self, key, weights):
        # Same as ExpertStore.remember, but sized with .nbytes, which Int4Weight has too
        size = sum(t.nbytes for t in weights)
        if size <= self.cache_bytes:
            while self.cached_bytes + size > self.cache_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= sum(t.nbytes for t in evicted)
            self.cache[key] = weights
            self.cached_bytes += size

    def part_tensor(self, block: bytearray, layer: int, name: str, part: dict):
        shape, type_name = self.parts[f"{layer}.{name}"]
        if type_name == "INT4_G32":
            return Int4Weight.from_buffer(block, part["offset"], shape)
        if type_name == "INT8_G16":
            return dequant_int8_g16(block, part["offset"], shape)
        raw = torch.frombuffer(block, dtype=torch.uint8, count=part["size"], offset=part["offset"])
        return dequantize(raw, type_name, shape)

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

    if arch == "qwen35moe":
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
    """Reverses what convert_hf_to_gguf.py does to Qwen3.5/3.6 tensors (Qwen3NextModel and
    _LinearAttentionVReorderBase there). t is float32."""
    if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
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

def hf_names(gguf_name: str, prefix: str, cfg: dict, tensors: dict):
    """Maps one llama.cpp tensor to [(Hugging Face name, row slice or None)]; [] for tensors the engine skips."""
    if gguf_name == "output.weight":
        return [("lm_head.weight", None)]
    if gguf_name in GLOBAL_NAMES:
        return [(prefix + GLOBAL_NAMES[gguf_name], None)]
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
    if rest.endswith(".bias") and rest not in LAYER_NAMES:
        # Biases sit next to their weight: blk.N.attn_q.bias -> ...q_proj.bias
        weight_name = LAYER_NAMES.get(rest.removesuffix(".bias") + ".weight")
        if weight_name:
            return [(p + weight_name.removesuffix(".weight") + ".bias", None)]
    if rest not in LAYER_NAMES:
        raise NotImplementedError(f"Don't know where GGUF tensor {gguf_name} goes")
    return [(p + LAYER_NAMES[rest], None)]

def load_weights(arch: str, prefix: str, cfg: dict, tensors: dict) -> dict:
    """Reads and dequantizes every non-expert tensor. Returns {Hugging Face name: tensor}, all resident in RAM."""
    weights = {}
    fds = {}
    try:
        for gguf_name, t in tensors.items():
            targets = hf_names(gguf_name, prefix, cfg, tensors)
            if not targets:
                continue
            if t.path not in fds:
                fds[t.path] = os.open(t.path, os.O_RDONLY)
            raw = bytearray(t.nbytes)
            if os.preadv(fds[t.path], [raw], t.offset) != t.nbytes:
                raise IOError(f"short read for {gguf_name} in {t.path}")
            for name, rows in targets:
                # Tensors that need the converter's changes undone go through float32; the rest (including the
                # big embedding and output tables) dequantize straight to bf16
                fix = arch == "qwen35moe" and (".linear_attn." in name or name.endswith(("norm.weight", "_gate.weight")))
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

def build_tokenizer(metadata: dict) -> Tokenizer:
    """Rebuilds the Hugging Face byte-level BPE tokenizer from the vocabulary and merges stored in the GGUF."""
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        raise NotImplementedError(f"tokenizer type {metadata.get('tokenizer.ggml.model')!r} is not supported "
                                  "(only byte-level BPE, 'gpt2')")
    pre = metadata.get("tokenizer.ggml.pre")
    if pre not in PRE_TOKENIZER_REGEXES:
        raise NotImplementedError(f"pre-tokenizer {pre!r} is not supported (supported: {', '.join(PRE_TOKENIZER_REGEXES)})")

    tokens = metadata["tokenizer.ggml.tokens"]
    merges = [tuple(m.split(" ", 1)) for m in metadata["tokenizer.ggml.merges"]]
    tokenizer = Tokenizer(models.BPE({t: i for i, t in enumerate(tokens)}, merges))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRE_TOKENIZER_REGEXES[pre]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    # Control tokens (<|im_start|>, ...) and user-defined ones (<think>, ...) match as whole strings before BPE
    types = metadata.get("tokenizer.ggml.token_type", [])
    tokenizer.add_special_tokens([AddedToken(tokens[i], special=True, normalized=False)
                                  for i, kind in enumerate(types) if kind == CONTROL_TOKEN])
    tokenizer.add_tokens([AddedToken(tokens[i], special=False, normalized=False)
                          for i, kind in enumerate(types) if kind == USER_DEFINED_TOKEN])
    return tokenizer

def load_engine(gguf_path: str, pack_dir: str = None, cache_gb: float = 0) -> MoEEngine:
    metadata, tensors = read_gguf(gguf_path)
    arch = metadata.get("general.architecture")
    if arch not in ENGINES:
        raise NotImplementedError(f"GGUF architecture {arch!r} is not supported (supported: {', '.join(ENGINES)})")
    engine_class = ENGINES[arch]
    cfg = build_config(metadata, tensors)

    tokenizer = build_tokenizer(metadata)
    vocab = tokenizer.get_vocab()
    eos_ids = {metadata[k] for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id") if k in metadata}
    eos_ids |= {vocab[t] for t in END_TOKENS if t in vocab}
    cfg["bos_token_id"] = metadata.get("tokenizer.ggml.bos_token_id", vocab.get("<|endoftext|>"))

    weights = load_weights(arch, engine_class.prefix, cfg, tensors)
    experts = GGUFExpertStore(pack_dir or default_pack_dir(gguf_path), int(cache_gb * 1024**3))
    return engine_class(cfg, eos_ids, weights, experts, tokenizer)
