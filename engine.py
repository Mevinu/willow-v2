import os
import math
import json
import time
import argparse
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

from main import EXPERT_RE, build_index_map

SAFETENSORS_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}

# Weights the text engine never uses: vision encoder and the multi-token-prediction head
SKIP_PREFIXES = ("model.visual.", "mtp.")

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)

def rope_parameters(cfg: dict) -> dict:
    # transformers 5.x saves "rope_parameters" (with rope_theta inside), 4.x saves "rope_theta" + "rope_scaling"
    params = dict(cfg.get("rope_parameters") or cfg.get("rope_scaling") or {})
    params.setdefault("rope_theta", cfg.get("rope_theta", 10000.0))
    params.setdefault("rope_type", params.get("type", "default"))
    return params

def compute_inv_freq(cfg: dict, head_dim: int):
    """Returns (inv_freq, attention_factor) for default or YaRN RoPE, matching transformers."""
    params = rope_parameters(cfg)
    base = params["rope_theta"]
    pos_freqs = base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    if params["rope_type"] == "default":
        return 1.0 / pos_freqs, 1.0
    if params["rope_type"] != "yarn":
        raise NotImplementedError(f"RoPE type {params['rope_type']!r} is not supported")

    factor = params["factor"]
    original_max_pos = params.get("original_max_position_embeddings") or cfg["max_position_embeddings"]
    beta_fast = params.get("beta_fast") or 32
    beta_slow = params.get("beta_slow") or 1

    def get_mscale(scale, mscale=1):
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

    attention_factor = params.get("attention_factor")
    if attention_factor is None:
        mscale, mscale_all_dim = params.get("mscale"), params.get("mscale_all_dim")
        if mscale and mscale_all_dim:
            attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
        else:
            attention_factor = get_mscale(factor)

    def correction_dim(num_rotations):
        return (head_dim * math.log(original_max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if params.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, head_dim - 1)
    if low == high:
        high += 0.001

    ramp = ((torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation_factor = 1 - ramp
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - extrapolation_factor) + (1.0 / pos_freqs) * extrapolation_factor
    return inv_freq, attention_factor

class ExpertStore:
    """Reads one expert's [gate | up | down] (or [gate_up | down]) block from the pack with a single aligned read.
    Recently used experts are kept in an LRU cache of up to cache_bytes."""

    def __init__(self, pack_dir: str, cache_bytes: int = 0):
        with open(os.path.join(pack_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        if "parts" not in self.manifest:
            raise ValueError(f"{pack_dir} was built by an older main.py; rerun main.py on the model to rebuild it")
        self.fd = os.open(os.path.join(pack_dir, "experts.pack"), os.O_RDONLY)
        self.parts = {
            name: (tuple(info["shape"]), SAFETENSORS_DTYPES[info["dtype"]])
            for name, info in self.manifest["parts"].items()
        }
        self.cache = OrderedDict()  # (layer, expert) -> (gate, up, down), least recently used first
        self.cache_bytes = cache_bytes
        self.cached_bytes = 0
        self.hits = 0
        self.reads = 0
        self.bytes_read = 0
        self.read_seconds = 0.0

    def load(self, layer: int, expert: int):
        """Returns (gate, up, down) weight matrices for one expert, from the cache or the pack."""
        key = (layer, expert)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        weights = self.read(layer, expert)
        size = sum(t.numel() * t.element_size() for t in weights)
        if size <= self.cache_bytes:
            while self.cached_bytes + size > self.cache_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= sum(t.numel() * t.element_size() for t in evicted)
            self.cache[key] = weights
            self.cached_bytes += size
        return weights

    def read(self, layer: int, expert: int):
        """Reads one expert from the pack."""
        entry = self.manifest["experts"][f"{layer}_{expert}"]
        start = time.perf_counter()
        block = bytearray(os.pread(self.fd, entry["read_length"], entry["file_offset"]))
        self.read_seconds += time.perf_counter() - start
        self.reads += 1
        self.bytes_read += len(block)

        weights = {}
        for name, part in entry["parts"].items():
            shape, dtype = self.parts[name]
            tensor = torch.frombuffer(block, dtype=dtype, count=math.prod(shape), offset=part["offset"])
            weights[name] = tensor.view(shape)
        if "gate_up" in weights:
            # Fused [2 * intermediate, hidden]: gate rows first, then up rows
            weights["gate"], weights["up"] = weights.pop("gate_up").chunk(2, dim=0)
        return weights["gate"], weights["up"], weights["down"]

    def close(self):
        os.close(self.fd)

class MoEEngine:
    """Qwen3-MoE. Subclasses override the pieces that differ for other architectures."""

    prefix = "model."            # where the decoder's weights live in the checkpoint
    zero_centered_norm = False   # RMSNorm scales by (1 + w) instead of w

    def __init__(self, model_dir: str, pack_dir: str = None, cache_gb: float = 0):
        with open(os.path.join(model_dir, "config.json")) as f:
            full_cfg = json.load(f)
        cfg = full_cfg.get("text_config", full_cfg)
        self.cfg = cfg
        self.setup(cfg)

        eos = cfg.get("eos_token_id")
        gen_cfg_path = os.path.join(model_dir, "generation_config.json")
        if os.path.exists(gen_cfg_path):
            with open(gen_cfg_path) as f:
                eos = json.load(f).get("eos_token_id", eos)
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

        # Everything except the routed experts stays resident in RAM. get_tensor only memory-maps the shard,
        # so clone() to really load it: otherwise macOS drops the pages under memory pressure and silently
        # re-reads them from disk on every token. The one exception is the embedding table when lm_head is
        # separate: each token reads a single row of it, so leaving it memory-mapped saves ~1 GB for free.
        index_map = build_index_map(model_dir)
        lazy = {self.prefix + "embed_tokens.weight"} if "lm_head.weight" in index_map else set()
        self.weights = {}
        for shard in sorted(set(index_map.values())):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if not EXPERT_RE.match(name) and not name.startswith(SKIP_PREFIXES):
                        tensor = f.get_tensor(name)
                        self.weights[name] = tensor if name in lazy else tensor.clone()
        if "lm_head.weight" not in self.weights:
            self.weights["lm_head.weight"] = self.weights[self.prefix + "embed_tokens.weight"]

        self.experts = ExpertStore(pack_dir or os.path.join(model_dir, "expert_pack"), int(cache_gb * 1024**3))
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.reset()

    def setup(self, cfg: dict):
        self.num_layers = cfg["num_hidden_layers"]
        self.layer_types = cfg.get("layer_types") or ["full_attention"] * self.num_layers
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or cfg["hidden_size"] // self.num_heads
        self.attn_output_gate = cfg.get("attn_output_gate", False)
        self.top_k = cfg["num_experts_per_tok"]
        self.norm_topk_prob = cfg.get("norm_topk_prob", False)
        self.eps = cfg["rms_norm_eps"]
        if cfg.get("mlp_only_layers") or cfg.get("decoder_sparse_step", 1) != 1:
            raise NotImplementedError("dense (mlp_only) layers are not supported yet")

        partial = rope_parameters(cfg).get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
        self.rotary_dim = int(self.head_dim * partial)
        self.inv_freq, self.attention_factor = compute_inv_freq(cfg, self.rotary_dim)

    def reset(self):
        self.kv_cache = [None] * self.num_layers
        self.position = 0
        self.timings = defaultdict(float)

    def w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        return F.linear(x, self.w(prefix + ".weight"), self.weights.get(prefix + ".bias"))

    def rms_norm(self, x: torch.Tensor, weight_name: str) -> torch.Tensor:
        weight = self.w(weight_name)
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.zero_centered_norm:
            return (x32 * (1.0 + weight.float())).type_as(x)
        return weight * x32.to(x.dtype)

    def rope(self, seq_len: int):
        positions = torch.arange(self.position, self.position + seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_factor, emb.sin() * self.attention_factor

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Only the first rotary_dim dims of each head are rotated (partial RoPE); the rest pass through
        x_rot, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        return torch.cat((x_rot * cos + rotate_half(x_rot) * sin, x_pass), dim=-1)

    def attention(self, x: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        p = f"{self.prefix}layers.{layer}.self_attn"
        seq_len = x.shape[0]
        q = self.linear(x, p + ".q_proj")
        gate = None
        if self.attn_output_gate:
            # q_proj emits [query | gate] per head; the gate scales the attention output
            q, gate = q.view(seq_len, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
            gate = gate.reshape(seq_len, self.num_heads * self.head_dim)
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = self.linear(x, p + ".k_proj").view(seq_len, self.num_kv_heads, self.head_dim)
        v = self.linear(x, p + ".v_proj").view(seq_len, self.num_kv_heads, self.head_dim)

        # Qwen3: per-head RMSNorm on q and k before RoPE
        q = self.rms_norm(q, p + ".q_norm.weight").transpose(0, 1)
        k = self.rms_norm(k, p + ".k_norm.weight").transpose(0, 1)
        v = v.transpose(0, 1)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        if self.kv_cache[layer] is not None:
            past_k, past_v = self.kv_cache[layer]
            k, v = torch.cat((past_k, k), dim=1), torch.cat((past_v, v), dim=1)
        self.kv_cache[layer] = (k, v)

        # Grouped-query attention: each KV head serves num_heads / num_kv_heads query heads
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=0)
        v = v.repeat_interleave(groups, dim=0)

        scores = (q @ k.transpose(1, 2)) * self.head_dim ** -0.5
        total_len = k.shape[1]
        causal = torch.ones(seq_len, total_len, dtype=torch.bool).tril(total_len - seq_len)
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (probs @ v).transpose(0, 1).reshape(seq_len, self.num_heads * self.head_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.linear(out, p + ".o_proj")

    def token_mixer(self, x: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return self.attention(x, layer, cos, sin)

    def tick(self, section: str, since: float) -> float:
        """Adds the time since `since` to a section's total; returns now, so calls can be chained."""
        now = time.perf_counter()
        self.timings[section] += now - since
        return now

    def moe(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        p = f"{self.prefix}layers.{layer}.mlp"
        t = time.perf_counter()
        router_logits = self.linear(x, p + ".gate")
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        t = self.tick("router", t)

        out = torch.zeros_like(x)
        # Load each needed expert once, then run every token routed to it
        for expert in selected.unique().tolist():
            token_idx, slot = (selected == expert).nonzero(as_tuple=True)
            t = time.perf_counter()
            gate_w, up_w, down_w = self.experts.load(layer, expert)
            t = self.tick("expert load", t)
            h = x[token_idx]
            h = F.linear(F.silu(F.linear(h, gate_w)) * F.linear(h, up_w), down_w)
            out.index_add_(0, token_idx, h * routing_weights[token_idx, slot, None])
            self.tick("expert compute", t)

        # Shared expert (Qwen2-MoE / Qwen3.5-MoE): a dense MLP every token uses, scaled by a sigmoid gate
        sp = p + ".shared_expert"
        if sp + ".gate_proj.weight" in self.weights:
            t = time.perf_counter()
            shared = self.linear(F.silu(self.linear(x, sp + ".gate_proj")) * self.linear(x, sp + ".up_proj"), sp + ".down_proj")
            out = out + torch.sigmoid(self.linear(x, p + ".shared_expert_gate")) * shared
            self.tick("shared expert", t)
        return out

    @torch.no_grad()
    def forward(self, token_ids: list) -> torch.Tensor:
        """Runs new tokens through the model, extending the caches. Returns logits for each token."""
        x = self.w(self.prefix + "embed_tokens.weight")[torch.tensor(token_ids)]
        cos, sin = self.rope(len(token_ids))
        for layer in range(self.num_layers):
            p = f"{self.prefix}layers.{layer}"
            t = time.perf_counter()
            x = x + self.token_mixer(self.rms_norm(x, p + ".input_layernorm.weight"), layer, cos, sin)
            self.tick(self.layer_types[layer].replace("_", " "), t)
            x = x + self.moe(self.rms_norm(x, p + ".post_attention_layernorm.weight"), layer)
        self.position += len(token_ids)
        t = time.perf_counter()
        logits = F.linear(self.rms_norm(x, self.prefix + "norm.weight"), self.w("lm_head.weight"))
        self.tick("lm_head", t)
        return logits

    def prompt_ids(self, prompt: str) -> list:
        return [self.cfg["bos_token_id"]] + self.tokenizer.encode(prompt, add_special_tokens=False).ids

    def io_snapshot(self) -> tuple:
        """Marks the start of a phase: resets the per-section timings and snapshots the I/O counters."""
        self.timings = defaultdict(float)
        e = self.experts
        return time.perf_counter(), e.bytes_read, e.read_seconds, e.hits, e.reads

    def record_phase(self, name: str, tokens: int, start: tuple):
        now = time.perf_counter()
        e = self.experts
        timings = dict(self.timings)
        timings["other"] = (now - start[0]) - sum(timings.values())
        self.stats[name] = {
            "tokens": tokens,
            "seconds": now - start[0],
            "bytes_read": e.bytes_read - start[1],
            "read_seconds": e.read_seconds - start[2],
            "cache_hits": e.hits - start[3],
            "disk_reads": e.reads - start[4],
            "timings": timings,
        }

    def generate(self, prompt: str, max_new_tokens: int = 100, stream: bool = False) -> list:
        """Greedy decoding. Returns prompt + generated token ids; timings land in self.stats."""
        self.reset()
        self.stats = {}
        ids = self.prompt_ids(prompt)
        num_prompt = len(ids)
        printed = ""
        start = self.io_snapshot()
        logits = self.forward(ids)
        self.record_phase("prefill", num_prompt, start)

        # Decode phase: each step forwards the previous token. The first token comes from the prefill logits,
        # so the timed decode tokens are the forward passes after it.
        start = self.io_snapshot()
        decode_steps = 0
        for _ in range(max_new_tokens):
            next_id = int(logits[-1].argmax())
            if next_id in self.eos_ids:
                break
            ids.append(next_id)
            if stream:
                # Decode the whole reply so multi-byte characters split across tokens print correctly
                text = self.tokenizer.decode(ids[num_prompt:])
                print(text[len(printed):], end="", flush=True)
                printed = text
            logits = self.forward([next_id])
            decode_steps += 1
        self.record_phase("decode", decode_steps, start)
        if stream:
            print()
        return ids

class Qwen35MoEEngine(MoEEngine):
    """Qwen3.5/3.6-MoE: hybrid Gated DeltaNet (linear attention) + gated full attention, plus a shared expert."""

    prefix = "model.language_model."
    zero_centered_norm = True

    def setup(self, cfg: dict):
        super().setup(cfg)
        self.norm_topk_prob = True  # Qwen3.5's router always renormalizes the top-k weights
        self.lin_k_heads = cfg["linear_num_key_heads"]
        self.lin_v_heads = cfg["linear_num_value_heads"]
        self.lin_k_dim = cfg["linear_key_head_dim"]
        self.lin_v_dim = cfg["linear_value_head_dim"]
        self.conv_kernel = cfg["linear_conv_kernel_dim"]
        self.think = False

    def reset(self):
        super().reset()
        self.conv_state = [None] * self.num_layers       # last (kernel - 1) conv inputs per linear layer
        self.recurrent_state = [None] * self.num_layers  # [v_heads, k_dim, v_dim] fp32 per linear layer

    def token_mixer(self, x: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.layer_types[layer] == "linear_attention":
            return self.linear_attention(x, layer)
        return self.attention(x, layer, cos, sin)

    def linear_attention(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        p = f"{self.prefix}layers.{layer}.linear_attn"
        seq_len = x.shape[0]
        key_dim = self.lin_k_heads * self.lin_k_dim
        value_dim = self.lin_v_heads * self.lin_v_dim

        mixed = self.linear(x, p + ".in_proj_qkv")  # [seq, 2 * key_dim + value_dim]
        z = self.linear(x, p + ".in_proj_z").view(seq_len, self.lin_v_heads, self.lin_v_dim)
        b = self.linear(x, p + ".in_proj_b")
        a = self.linear(x, p + ".in_proj_a")

        # Causal depthwise conv over time, continuing from the previous (kernel - 1) inputs
        conv_w = self.w(p + ".conv1d.weight")  # [channels, 1, kernel]
        state = self.conv_state[layer]
        if state is None:
            state = mixed.new_zeros(self.conv_kernel - 1, mixed.shape[1])
        window = torch.cat((state, mixed))
        self.conv_state[layer] = window[-(self.conv_kernel - 1):]
        # Same result as F.conv1d(groups=channels), written as kernel-size shifted multiply-adds:
        # out[t, c] = sum_k window[t + k, c] * w[c, k]. F.conv1d with 8192 groups takes ~88 ms on CPU, this ~0.03 ms.
        conv_w, window = conv_w[:, 0, :].float(), window.float()
        conv = sum(window[k:k + seq_len] * conv_w[:, k] for k in range(self.conv_kernel))
        mixed = F.silu(conv.to(mixed.dtype))

        q, k, v = mixed.split([key_dim, key_dim, value_dim], dim=-1)
        q = q.reshape(seq_len, self.lin_k_heads, self.lin_k_dim)
        k = k.reshape(seq_len, self.lin_k_heads, self.lin_k_dim)
        v = v.reshape(seq_len, self.lin_v_heads, self.lin_v_dim)
        if self.lin_v_heads > self.lin_k_heads:
            q = q.repeat_interleave(self.lin_v_heads // self.lin_k_heads, dim=1)
            k = k.repeat_interleave(self.lin_v_heads // self.lin_k_heads, dim=1)

        beta = b.sigmoid()
        decay = -self.w(p + ".A_log").float().exp() * F.softplus(a.float() + self.w(p + ".dt_bias"))
        out = self.gated_delta_rule(q, k, v, decay, beta, layer).to(x.dtype)

        # Gated RMSNorm per head: norm(out) * w * silu(z)
        o32 = out.float()
        o32 = o32 * torch.rsqrt(o32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (self.w(p + ".norm.weight") * o32.to(out.dtype) * F.silu(z.float())).to(out.dtype)
        return self.linear(out.reshape(seq_len, value_dim), p + ".out_proj")

    def gated_delta_rule(self, q, k, v, decay, beta, layer: int) -> torch.Tensor:
        """Token-by-token gated delta rule. The state S [heads, k_dim, v_dim] is a fast-weight memory:
        each step decays it, then writes the part of v that S @ k does not already predict."""
        q, k, v, beta = (t.float() for t in (q, k, v, beta))
        q = l2norm(q) * self.lin_k_dim ** -0.5
        k = l2norm(k)
        state = self.recurrent_state[layer]
        if state is None:
            state = torch.zeros(self.lin_v_heads, self.lin_k_dim, self.lin_v_dim)
        outputs = []
        for t in range(q.shape[0]):
            state = state * decay[t].exp()[:, None, None]
            predicted = torch.einsum("hkv,hk->hv", state, k[t])
            delta = (v[t] - predicted) * beta[t][:, None]
            state = state + k[t][:, :, None] * delta[:, None, :]
            outputs.append(torch.einsum("hkv,hk->hv", state, q[t]))
        self.recurrent_state[layer] = state
        return torch.stack(outputs)

    def prompt_ids(self, prompt: str) -> list:
        # Qwen chat format; an empty think block turns thinking off
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        text += "<think>\n" if self.think else "<think>\n\n</think>\n\n"
        return self.tokenizer.encode(text, add_special_tokens=False).ids

ENGINES = {"qwen3_moe": MoEEngine, "qwen3_5_moe": Qwen35MoEEngine}

def load_engine(model_dir: str, pack_dir: str = None, cache_gb: float = 0) -> MoEEngine:
    with open(os.path.join(model_dir, "config.json")) as f:
        model_type = json.load(f).get("model_type")
    if model_type not in ENGINES:
        raise NotImplementedError(f"model_type {model_type!r} is not supported (supported: {', '.join(ENGINES)})")
    return ENGINES[model_type](model_dir, pack_dir, cache_gb)

def main():
    parser = argparse.ArgumentParser(description="Run a MoE model, streaming experts from its expert pack")
    parser.add_argument("model_dir", help="Model folder (the one main.py was run on)")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="upper limit; generation stops earlier when the model emits an end token")
    parser.add_argument("--pack-dir", default=None, help="Default: <model_dir>/expert_pack")
    parser.add_argument("--think", action="store_true", help="Let chat models think before answering")
    parser.add_argument("--cache-gb", type=float, default=0.0,
                        help="RAM for caching recently used experts (default off: on a 16 GB Mac the OS file cache "
                             "does better, and an explicit cache causes memory pressure)")
    args = parser.parse_args()

    start = time.perf_counter()
    engine = load_engine(args.model_dir, args.pack_dir, args.cache_gb)
    load_seconds = time.perf_counter() - start
    engine.think = args.think
    if isinstance(engine, Qwen35MoEEngine):
        print(f"> {args.prompt}\n")
    else:
        print(args.prompt, end="", flush=True)
    engine.generate(args.prompt, args.max_new_tokens, stream=True)

    print(f"\n--- stats ---\nload:     {load_seconds:.1f}s")
    for name, s in engine.stats.items():
        if s["tokens"] == 0:
            continue
        gib = s["bytes_read"] / 1024**3
        io_share = s["read_seconds"] / s["seconds"] * 100
        io_speed = gib / s["read_seconds"] if s["read_seconds"] else 0
        lookups = s["cache_hits"] + s["disk_reads"]
        hit_rate = s["cache_hits"] / lookups * 100 if lookups else 0
        print(f"{name + ':':9} {s['tokens']:4d} tokens in {s['seconds']:6.1f}s = {s['tokens'] / s['seconds']:6.2f} tok/s | "
              f"cache hits {hit_rate:3.0f}% | {gib:6.2f} GiB read ({gib / s['tokens']:.2f} GiB/token) | "
              f"disk {io_share:3.0f}% of time at {io_speed:.1f} GiB/s")
    cache = engine.experts
    print(f"cache:    {cache.cached_bytes / 1024**3:.2f} / {args.cache_gb:.2f} GiB used, {len(cache.cache)} experts held")

    decode = engine.stats.get("decode")
    if decode and decode["tokens"]:
        print("\ndecode time per token, by section:")
        for section, seconds in sorted(decode["timings"].items(), key=lambda kv: -kv[1]):
            ms = seconds / decode["tokens"] * 1000
            print(f"  {section:18s} {ms:8.1f} ms  ({seconds / decode['seconds'] * 100:4.1f}%)")

if __name__ == "__main__":
    main()
