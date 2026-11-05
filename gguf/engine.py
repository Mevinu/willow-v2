"""Runs GGUF models (as made by llama.cpp's convert_hf_to_gguf.py) on the same model code as the safetensors
engine. The non-expert weights are dequantized to bf16 at load (bf16 matmuls run faster than torch's int4 kernel
for them on Apple Silicon) and renamed to their Hugging Face names, undoing the changes llama.cpp's converter
makes. The routed experts stay quantized in the pack; int4 ones (all of them, for a Q4_K/Q6_K model) run on
torch's int4 kernel, and other types are dequantized as each one is read."""
import fcntl
import json
import math
import mmap
import os
import re
import sys
import time
from collections import OrderedDict

import torch
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from safetensor.compiler import PAGE_ALIGNMENT, align_up
from safetensor.engine import ExpertStore, MoEEngine, Qwen35MoEEngine, expert_weights
from .compiler import EXPERT_RE, default_pack_dir, num_decoder_layers
from .quants import PACK_LAYOUTS, Int4Weight, dequant_int8_g16, dequantize, int4_nbytes
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
USAGE_FILE = "expert_usage.json"  # the expert usage table, next to manifest.json in the pack
END_TOKENS = ("<|im_end|>", "<|endoftext|>")  # Qwen's generation_config.json stops on both

class GGUFExpertStore(ExpertStore):
    """Reads experts from a pack built from a GGUF file. INT4_G32 parts stay 4-bit (Int4Weight); the rest are
    dequantized to bf16 on the read threads.

    With cache_bytes, recently used experts are kept in a slab: memory allocated once and cut into fixed slots, one
    expert each, evicted least recently used first. Its size is exact (the weights live in the slots, and nothing
    else is kept alive), and the OS file cache is turned off for the pack so the same bytes aren't held twice.
    Layers whose slots are the same size share a slab, with the same number of slots per layer. A slot holds the
    expert as it is computed with: int4 parts as stored, everything else as bf16, so a hit costs nothing. Layers
    that are all int4 (every layer, in a pack of a Q4_K/Q6_K model) are read straight into their slot.
    Only the main thread changes the cache; a read thread just fills the one slot it was given."""

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
        if "INT8_G16" in types:
            print(f"note: {pack_dir} stores Q6_K experts as int8, which is dequantized as each expert is read; rerun "
                  "`main.py pack` on this model to store them as int4 (smaller, and run without dequantizing)")
        self.slabs = {}  # layer -> its slab group (shared by layers with the same slot layout)
        if cache_bytes:
            self.build_slabs(cache_bytes)
        self.load_usage(pack_dir)

    # Expert usage table: how many tokens each (layer, expert) has been routed to, summed over every run on this
    # pack. Kept in RAM as a [num_layers, num_experts] count tensor, saved to USAGE_FILE in the pack after a run.

    def load_usage(self, pack_dir: str):
        self.usage_path = os.path.join(pack_dir, USAGE_FILE)
        shape = (self.manifest["num_layers"], self.manifest["num_experts"])
        self.usage = torch.zeros(shape, dtype=torch.int64)
        self.usage_runs = 0
        try:
            with open(self.usage_path) as f:
                saved = json.load(f)
        except FileNotFoundError:
            return
        counts = torch.tensor(saved["counts"], dtype=torch.int64)
        if counts.shape != shape:
            print(f"note: {self.usage_path} is for a different model shape; starting a new usage table")
            return
        self.usage, self.usage_runs = counts, saved.get("runs", 0)

    def record_usage(self, layer: int, selected: torch.Tensor):
        self.usage[layer] += torch.bincount(selected.flatten(), minlength=self.usage.shape[1])

    def save_usage(self):
        """Writes the table (to a temp file, then renamed over the old one, so an interrupted write can't lose it)."""
        self.usage_runs += 1
        tmp = self.usage_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"runs": self.usage_runs, "counts": self.usage.tolist()}, f)
        os.replace(tmp, self.usage_path)

    def most_used(self, layer: int, n: int = None) -> list:
        """The layer's experts, most used first (the top n, if given), as (expert, count) pairs."""
        counts, order = self.usage[layer].sort(descending=True, stable=True)
        return list(zip(order.tolist(), counts.tolist()))[:n]

    def build_slabs(self, cache_bytes: int):
        num_experts = self.manifest["num_experts"]
        self.slot_parts = {}  # layer -> {part: (offset in slot, size in slot)}
        self.direct = {}      # layer -> whether its slot is a byte-for-byte copy of the pack block
        layouts = {}          # (direct, slot size) -> layers
        for layer in range(self.manifest["num_layers"]):
            entries = [self.manifest["experts"][f"{layer}_{e}"] for e in range(num_experts)]
            parts, offset = {}, 0
            for name in entries[0]["parts"]:
                shape, type_name = self.parts[f"{layer}.{name}"]
                size = int4_nbytes(shape) if type_name == "INT4_G32" else math.prod(shape) * 2
                parts[name] = (offset, size)
                offset += size
            direct = all(self.parts[f"{layer}.{name}"][1] == "INT4_G32" for name in parts)
            if direct:
                assert all(e["parts"][n]["offset"] == parts[n][0] for e in entries for n in parts)
                offset = max(e["read_length"] for e in entries)
            self.slot_parts[layer], self.direct[layer] = parts, direct
            layouts.setdefault((direct, align_up(offset, PAGE_ALIGNMENT)), []).append(layer)

        per_layer = cache_bytes // sum(slot * len(layers) for (_, slot), layers in layouts.items())
        if per_layer * min(len(layers) for layers in layouts.values()) < 2 * self.read_ahead:
            print("note: --cache-gb is too small to be useful; cache off")
            self.cache_bytes = 0
            return
        for (_, slot), layers in layouts.items():
            count = per_layer * len(layers)
            group = {"slab": mmap.mmap(-1, count * slot), "slot": slot, "free": list(range(count)),
                     "lru": OrderedDict()}  # lru: (layer, expert) -> slot number, least recently used first
            for layer in layers:
                self.slabs[layer] = group
        if sys.platform == "darwin":
            fcntl.fcntl(self.fd, getattr(fcntl, "F_NOCACHE", 48), 1)
        self.cache_bytes = sum(g["slot"] * len(g["free"]) for g in {id(g): g for g in self.slabs.values()}.values())

    # ExpertStore.load_many and prefetch call these hooks; with a slab, a read goes straight into a free slot

    def lookup(self, key):
        if not self.slabs:
            return super().lookup(key)
        weights = self.cache.get(key)
        if weights is not None:
            self.slabs[key[0]]["lru"].move_to_end(key)
            self.hits += 1
        return weights

    def start_read(self, layer: int, expert: int, prefetch: bool = False):
        if not self.slabs:
            return super().start_read(layer, expert, prefetch)
        group = self.slabs[layer]
        slot = self.take_slot(group)
        if slot is None:
            # Every slot is busy (a huge prefill batch): read without caching, unless it's only a guess
            return None if prefetch else (self.pool.submit(self.read, layer, expert), None)
        return self.pool.submit(self.fill, layer, expert, group, slot), slot

    def finish_read(self, key, weights, slot):
        if not self.slabs:
            return super().finish_read(key, weights, slot)
        if slot is not None:
            group = self.slabs[key[0]]
            group["lru"][key] = slot
            self.cache[key] = weights
            self.cached_bytes += group["slot"]

    def take_slot(self, group: dict):
        """A free slot, else the least recently used one not handed out by load_many (None if there is none).
        Slots being filled are in neither the free list nor the LRU, so they are never taken."""
        if group["free"]:
            return group["free"].pop()
        victim = next((key for key in group["lru"] if key not in self.busy), None)
        if victim is None:
            return None
        del self.cache[victim]
        self.cached_bytes -= group["slot"]
        return group["lru"].pop(victim)

    def fill(self, layer: int, expert: int, group: dict, slot: int):
        """Reads one expert into its slot and returns its weights, viewing the slot. Runs on the thread pool."""
        entry = self.manifest["experts"][f"{layer}_{expert}"]
        slab, base = group["slab"], slot * group["slot"]
        with memoryview(slab) as view:
            if self.direct[layer]:
                self.pread(view[base:base + entry["read_length"]], layer, expert, entry)
            else:
                block = bytearray(entry["read_length"])
                self.pread(block, layer, expert, entry)
                for name, part in entry["parts"].items():
                    offset, size = self.slot_parts[layer][name]
                    if self.parts[f"{layer}.{name}"][1] == "INT4_G32":
                        view[base + offset:base + offset + size] = memoryview(block)[part["offset"]:part["offset"] + size]
                    else:
                        self.slot_tensor(slab, base, layer, name).copy_(self.part_tensor(block, layer, name, part))
        return expert_weights({name: self.slot_tensor(slab, base, layer, name) for name in self.slot_parts[layer]})

    def slot_tensor(self, slab: mmap.mmap, base: int, layer: int, name: str):
        shape, type_name = self.parts[f"{layer}.{name}"]
        offset = base + self.slot_parts[layer][name][0]
        if type_name == "INT4_G32":
            return Int4Weight.from_buffer(slab, offset, shape)
        return torch.frombuffer(slab, dtype=torch.bfloat16, count=math.prod(shape), offset=offset).view(shape)

    def pread(self, buf, layer: int, expert: int, entry: dict):
        """os.preadv into buf, with the same I/O accounting as ExpertStore.read."""
        with self.lock:
            if self.in_flight == 0:
                self.busy_since = time.perf_counter()
            self.in_flight += 1
        try:
            got = os.preadv(self.fd, [buf], entry["file_offset"])
        finally:
            with self.lock:
                self.in_flight -= 1
                if self.in_flight == 0:
                    self.read_seconds += time.perf_counter() - self.busy_since
        if got != len(buf):
            raise IOError(f"short read for expert {layer}_{expert}: {got} of {len(buf)} bytes")
        with self.lock:
            self.reads += 1
            self.bytes_read += len(buf)

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

DEFAULT_CACHE_GB = 8.0   # expert cache when --cache-gb isn't given...
RAM_HEADROOM_GB = 4.0    # ...shrunk so the weights + cache leave this much RAM for macOS, the KV cache and buffers

def auto_cache_gb(weights: dict) -> float:
    """DEFAULT_CACHE_GB, or less if the machine can't spare it next to the resident weights."""
    total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    weights_gb = sum(t.nbytes for t in weights.values()) / 1024**3
    return max(0.0, min(DEFAULT_CACHE_GB, total_gb - weights_gb - RAM_HEADROOM_GB))

def load_engine(gguf_path: str, pack_dir: str = None, cache_gb: float = None) -> MoEEngine:
    """cache_gb None (no --cache-gb) sizes the expert cache automatically: see auto_cache_gb."""
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
    if cache_gb is None:
        cache_gb = auto_cache_gb(weights)
        print(f"expert cache: {cache_gb:.1f} GiB (auto; set it with --cache-gb)")
    experts = GGUFExpertStore(pack_dir or default_pack_dir(gguf_path), int(cache_gb * 1024**3))
    return engine_class(cfg, eos_ids, weights, experts, tokenizer)
