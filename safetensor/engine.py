import os
import math
import json
import time
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

from .compiler import EXPERT_RE, build_index_map

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

# The small steps of each layer, as plain functions of tensors. On the GPU each op costs ~0.01-0.02 ms to issue
# however small it is, and a layer has ~100 of them, so there they are compiled with torch.compile, which fuses
# each into a few GPU kernels (see Ops). On the CPU they run as they are.

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool) -> torch.Tensor:
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    if zero_centered:
        return (x32 * (1.0 + weight.float())).type_as(x)
    return weight * x32.to(x.dtype)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    # Only the first rotary_dim dims of each head are rotated (partial RoPE); the rest pass through
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x_rot * cos + rotate_half(x_rot) * sin, x_pass), dim=-1)

def attention_core(q, k, v, mask, gate, groups: int, scale: float) -> torch.Tensor:
    """q [heads, seq, dim], k and v [kv_heads, capacity, dim], mask [seq, capacity] -> [seq, heads * dim]."""
    # Grouped-query attention: each KV head serves num_heads / num_kv_heads query heads
    k = k.repeat_interleave(groups, dim=0)
    v = v.repeat_interleave(groups, dim=0)
    scores = (q @ k.transpose(1, 2)) * scale + mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = (probs @ v).transpose(0, 1).reshape(q.shape[1], q.shape[0] * q.shape[2])
    if gate is not None:
        out = out * torch.sigmoid(gate)
    return out

def route(router_logits: torch.Tensor, top_k: int, normalize: bool):
    """-> (routing weights [tokens, top_k] in router_logits' dtype, selected experts [tokens, top_k])"""
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected = torch.topk(routing_weights, top_k, dim=-1)
    if normalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.to(router_logits.dtype), selected

def expert_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """A routed expert's fused [gate | up] output -> the input of its down matmul."""
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up

def shared_expert(x, gate_w, up_w, down_w, shared_gate_w) -> torch.Tensor:
    """Qwen2-MoE / Qwen3.5-MoE: a dense MLP every token uses, scaled by a sigmoid gate."""
    h = F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)
    return torch.sigmoid(F.linear(x, shared_gate_w)) * h

def linear_attention_step(mixed, z, b, a, conv_state, conv_w, A_log, dt_bias, norm_w, state,
                          k_heads: int, v_heads: int, k_dim: int, v_dim: int, eps: float):
    """One decode token of Gated DeltaNet, after its input projections: the same as Qwen35MoEEngine's
    linear_attention + gated_delta_rule for seq_len 1. Returns (output [1, v_heads * v_dim] for out_proj,
    new conv state, new recurrent state)."""
    key_dim = k_heads * k_dim
    window = torch.cat((conv_state, mixed))  # [kernel, channels]
    conv = (window.float() * conv_w[:, 0, :].float().t()).sum(0, keepdim=True)
    mixed_act = F.silu(conv.to(mixed.dtype))
    q, k, v = mixed_act.split([key_dim, key_dim, v_heads * v_dim], dim=-1)
    q, k, v = q.reshape(k_heads, k_dim), k.reshape(k_heads, k_dim), v.reshape(v_heads, v_dim)
    if v_heads > k_heads:
        q = q.repeat_interleave(v_heads // k_heads, dim=0)
        k = k.repeat_interleave(v_heads // k_heads, dim=0)
    beta = b[0].sigmoid().float()
    decay = -A_log.float().exp() * F.softplus(a[0].float() + dt_bias)
    q = l2norm(q.float()) * k_dim ** -0.5
    k = l2norm(k.float())
    state = state * decay.exp()[:, None, None]
    predicted = torch.einsum("hkv,hk->hv", state, k)
    delta = (v.float() - predicted) * beta[:, None]
    state = state + k[:, :, None] * delta[:, None, :]
    out = torch.einsum("hkv,hk->hv", state, q).to(mixed.dtype)
    # Gated RMSNorm per head: norm(out) * w * silu(z)
    o32 = out.float()
    o32 = o32 * torch.rsqrt(o32.pow(2).mean(-1, keepdim=True) + eps)
    out = (norm_w * o32.to(out.dtype) * F.silu(z.view(v_heads, v_dim).float())).to(out.dtype)
    return out.reshape(1, v_heads * v_dim), window[1:], state

class Ops:
    """The functions above, run as they are, or (with compile set, for the GPU) compiled with torch.compile while
    decoding. Compiled for fixed shapes only (torch 2.8's GPU compiler fails on shapes that vary), so a prefill
    batch, whose length varies, runs them as they are; decode steps all have the same shapes (the KV cache grows
    in KV_BLOCK steps, each compiled once)."""
    NAMES = ("rms_norm", "apply_rope", "attention_core", "route", "expert_gate", "shared_expert",
             "linear_attention_step")

    def __init__(self, compile: bool):
        self.compiled = compile
        self.decoding = False  # set by MoEEngine.forward: compiled versions are used only while this is True
        if compile:
            import logging
            import torch._dynamo
            # Hides inductor's "Not enough SMs to use max_autotune_gemm mode" warning (harmless; it's about CUDA)
            logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
            for limit in ("recompile_limit", "cache_size_limit"):  # one compile per KV-cache size, at most
                if hasattr(torch._dynamo.config, limit):
                    setattr(torch._dynamo.config, limit, 64)
        for name in self.NAMES:
            fn = globals()[name]
            setattr(self, name, self.dispatch(fn, torch.compile(fn, dynamic=False)) if compile else fn)

    def dispatch(self, eager, compiled):
        return lambda *args: (compiled if self.decoding else eager)(*args)

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

PREFETCH_MAX_IN_FLIGHT = 4  # default ExpertStore.prefetch_max_in_flight
# The KV cache grows in steps of this many positions, and attention always runs over the whole allocated length
# (the unused end masked off). On the GPU every new tensor shape costs a fresh setup of each op (~11 ms per
# attention layer when the cache grew by one position per token), so the shapes only change every KV_BLOCK tokens.
KV_BLOCK = 256

def expert_weights(parts: dict) -> tuple:
    """An expert's parts by name -> (gate_up, down) when gate and up are fused into one
    [2 * intermediate, hidden] matrix (gate rows first), else (gate, up, down)."""
    if "gate_up" in parts:
        return parts["gate_up"], parts["down"]
    return parts["gate"], parts["up"], parts["down"]

class ExpertStore:
    """Reads one expert's [gate | up | down] (or [gate_up | down]) block from the pack with a single aligned read.
    A layer's experts are read in parallel on a thread pool (os.preadv releases the GIL), so the SSD has many
    requests in flight and later reads finish while earlier experts are being computed.
    Recently used experts are kept in an LRU cache of up to cache_bytes."""

    # prefetch does nothing while this many reads are already running (None: never skip). The pool runs reads in
    # the order they were submitted, so even uncapped, guesses only start once the current layer's reads have.
    prefetch_max_in_flight = PREFETCH_MAX_IN_FLIGHT

    pack_format = "safetensors"  # the manifest's "format"; packs from before the field existed are safetensors

    def __init__(self, pack_dir: str, cache_bytes: int = 0, read_threads: int = 8, read_ahead: int = 16):
        with open(os.path.join(pack_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        if "parts" not in self.manifest:
            raise ValueError(f"{pack_dir} was built by an older main.py; rerun `main.py pack` on the model to rebuild it")
        if self.manifest.get("format", "safetensors") != self.pack_format:
            raise ValueError(f"{pack_dir} is a {self.manifest.get('format', 'safetensors')} pack, "
                             f"but the model is {self.pack_format}; rerun `main.py pack` on this model")
        self.fd = os.open(os.path.join(pack_dir, "experts.pack"), os.O_RDONLY)
        self.parts = {name: (tuple(info["shape"]), info["dtype"]) for name, info in self.manifest["parts"].items()}
        self.cache = OrderedDict()  # (layer, expert) -> (gate, up, down), least recently used first
        self.cache_bytes = cache_bytes
        self.cached_bytes = 0
        self.hits = 0
        self.reads = 0
        self.bytes_read = 0
        self.read_seconds = 0.0  # wall time with at least one read in flight (not summed per thread)
        self.pool = ThreadPoolExecutor(read_threads)
        self.read_ahead = read_ahead  # max experts requested but not yet consumed (bounds memory during prefill)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.busy_since = 0.0
        self.busy = set()     # (layer, expert) handed out by load_many and not yet consumed
        self.prefetched = {}  # (layer, expert) -> (future, token): prefetch reads not yet picked up
        self.prefetches = 0       # prefetch reads started
        self.prefetch_used = 0    # of those, ones load_many then asked for

    def load_many(self, layer: int, experts: list):
        """Starts loading a layer's experts and returns an iterator of (expert, weights), where weights is
        (gate_up, down) or (gate, up, down). The reads for misses are submitted before this returns (up to
        read_ahead of them; for a decode step, all), so the SSD works on them while the caller computes the
        cache hits, which come first. Misses then come in the order their reads finish."""
        keys = [(layer, expert) for expert in experts]
        self.collect_prefetches(layer, exclude=set(keys))
        self.busy.update(keys)  # their cache entries must not be evicted until they have been consumed
        hits, misses = [], []
        for key in keys:
            weights = self.lookup(key)
            if weights is None and key in self.prefetched and self.prefetched[key][0].done():
                future, token = self.prefetched.pop(key)  # a prefetch that has already finished: as good as a hit
                weights = self.finish_read(key, future.result(), token)
                self.prefetch_used += 1
            if weights is not None:
                hits.append((key[1], weights))
            else:
                misses.append(key[1])

        reads = {}  # future -> (expert, token for finish_read)
        def start(expert):
            key = (layer, expert)
            if key in self.prefetched:  # prefetched, still reading
                self.prefetch_used += 1
                future, token = self.prefetched.pop(key)
            else:
                future, token = self.start_read(layer, expert)
            reads[future] = (expert, token)

        pending = iter(misses)
        for expert in islice(pending, self.read_ahead):
            start(expert)
        return self.results(layer, keys, hits, reads, pending, start)

    def results(self, layer: int, keys: list, hits: list, reads: dict, pending, start):
        try:
            for expert, weights in hits:
                yield expert, weights
                self.busy.discard((layer, expert))
            while reads:
                done, _ = wait(reads, return_when=FIRST_COMPLETED)
                for future in done:
                    expert, token = reads.pop(future)
                    weights = self.finish_read((layer, expert), future.result(), token)
                    following = next(pending, None)
                    if following is not None:
                        start(following)
                    yield expert, weights
                    self.busy.discard((layer, expert))
        finally:
            self.busy.difference_update(keys)

    def prefetch(self, layer: int, experts: list):
        """Starts reading experts that will probably be needed soon, without waiting for them. load_many reuses
        these reads (they are never read twice); ones it never asks for go into the cache once they finish, or
        are dropped if there is no cache. Skipped while the current layer still has prefetch_max_in_flight or
        more reads in flight, so guesses never hold up reads that are actually needed."""
        if self.prefetch_max_in_flight is not None and self.in_flight >= self.prefetch_max_in_flight:
            return
        for expert in experts:
            key = (layer, expert)
            if key in self.cache or key in self.prefetched:
                continue
            started = self.start_read(layer, expert, prefetch=True)
            if started is None:
                break
            self.prefetched[key] = started
            self.prefetches += 1

    def collect_prefetches(self, layer: int, exclude: set = frozenset()):
        """Deals with prefetches load_many isn't asking for, as it starts on layer: finished ones go into the cache.
        Without a cache, a guess is only worth anything to the layer it was made for, so guesses for this layer
        or earlier are dropped (cancelled, if their read hasn't started). Runs on the main thread."""
        for key, (future, token) in list(self.prefetched.items()):
            if key in exclude:
                continue
            if not self.cache_bytes and key[0] <= layer:
                del self.prefetched[key]
                future.cancel()
            elif future.done():
                del self.prefetched[key]
                self.finish_read(key, future.result(), token)

    def lookup(self, key):
        """Returns a cached expert's weights (marking it most recently used), or None."""
        weights = self.cache.get(key)
        if weights is not None:
            self.cache.move_to_end(key)
            self.hits += 1
        return weights

    def start_read(self, layer: int, expert: int, prefetch: bool = False):
        """Submits a read to the pool. Returns (future, token passed to finish_read), or None to skip a prefetch."""
        return self.pool.submit(self.read, layer, expert), None

    def finish_read(self, key, weights, token):
        """Called on the main thread once a read is done, with what the read returned. Returns the weights."""
        self.remember(key, weights)
        return weights

    def remember(self, key, weights):
        size = sum(t.numel() * t.element_size() for t in weights)
        if size <= self.cache_bytes:
            while self.cached_bytes + size > self.cache_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= sum(t.numel() * t.element_size() for t in evicted)
            self.cache[key] = weights
            self.cached_bytes += size

    def read(self, layer: int, expert: int):
        """Reads one expert from the pack. Runs on the thread pool."""
        entry = self.manifest["experts"][f"{layer}_{expert}"]
        block = bytearray(entry["read_length"])
        with self.lock:
            if self.in_flight == 0:
                self.busy_since = time.perf_counter()
            self.in_flight += 1
        try:
            # Reads straight into block (os.pread would return bytes that then need copying)
            got = os.preadv(self.fd, [block], entry["file_offset"])
        finally:
            with self.lock:
                self.in_flight -= 1
                if self.in_flight == 0:
                    self.read_seconds += time.perf_counter() - self.busy_since
        if got != len(block):
            raise IOError(f"short read for expert {layer}_{expert}: {got} of {len(block)} bytes")
        with self.lock:
            self.reads += 1
            self.bytes_read += len(block)

        weights = {name: self.part_tensor(block, layer, name, part) for name, part in entry["parts"].items()}
        return expert_weights(weights)

    def part_tensor(self, block: bytearray, layer: int, name: str, part: dict) -> torch.Tensor:
        """Turns one part's bytes inside a read block into a weight tensor (no copy: it views the block)."""
        shape, dtype = self.parts[name]
        tensor = torch.frombuffer(block, dtype=SAFETENSORS_DTYPES[dtype], count=math.prod(shape), offset=part["offset"])
        return tensor.view(shape)

    def record_usage(self, layer: int, selected: torch.Tensor):
        """Called by moe with the experts each token was routed to. The GGUF store counts them; this one doesn't."""

    def save_usage(self):
        """Called once a run is done, to save what record_usage counted."""

    def close(self):
        self.pool.shutdown()
        os.close(self.fd)

class MoEEngine:
    """Qwen3-MoE. Subclasses override the pieces that differ for other architectures."""

    prefix = "model."            # where the decoder's weights live in the checkpoint
    zero_centered_norm = False   # RMSNorm scales by (1 + w) instead of w
    prefetch_top_k = 8           # next-layer experts to start reading early while decoding (0: off)

    def __init__(self, cfg: dict, eos_ids: set, weights: dict, experts: ExpertStore, tokenizer: Tokenizer,
                 device: str = "cpu"):
        """cfg is the (text) config.json, and weights maps Hugging Face tensor names to every non-expert tensor.
        load_engine builds these from a safetensors folder, gguf.engine.load_engine from a .gguf file.
        device runs everything except the experts there (the experts run where their weights are): "mps" moves
        the non-expert weights to the GPU, freeing their CPU copies, and keeps the hidden state there."""
        self.cfg = cfg
        self.setup(cfg)
        self.eos_ids = eos_ids
        self.device = torch.device(device)
        self.ops = Ops(compile=self.device.type == "mps")
        self.sync_timings = False  # wait for the GPU at every section boundary, so the per-section times are real
        self.min_expert_weight = 0.0  # skip routed experts below this routing weight (0: off; see skip_small_experts)
        self.experts_routed = self.experts_skipped = 0  # while decoding, when skipping is on
        self.keep_alive = []       # CPU sources of non-blocking GPU copies, kept until the GPU has run them
        if self.device.type != "cpu":
            moved = {}  # tied tensors (lm_head = embed_tokens) are moved once
            for name in list(weights):
                tensor = weights[name]
                if id(tensor) not in moved:
                    moved[id(tensor)] = tensor.to(self.device)
                weights[name] = moved[id(tensor)]
            del moved, tensor
        self.weights = weights
        if "lm_head.weight" not in self.weights:
            self.weights["lm_head.weight"] = self.weights[self.prefix + "embed_tokens.weight"]
        self.experts = experts
        self.tokenizer = tokenizer
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
        self.kv_cache = [None] * self.num_layers  # per layer: (keys, values), each [kv_heads, capacity, head_dim]
        self.mask_key = None
        self.position = 0
        self.timings = defaultdict(float)

    def w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        return F.linear(x, self.w(prefix + ".weight"), self.weights.get(prefix + ".bias"))

    def rms_norm(self, x: torch.Tensor, weight_name: str) -> torch.Tensor:
        return self.ops.rms_norm(x, self.w(weight_name), self.eps, self.zero_centered_norm)

    def rope(self, seq_len: int):
        positions = torch.arange(self.position, self.position + seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return self.to_device(emb.cos() * self.attention_factor), self.to_device(emb.sin() * self.attention_factor)

    def to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Copies a CPU tensor to self.device without waiting (a waiting copy to the GPU waits for all queued GPU
        work), keeping the source alive until the GPU's next wait."""
        if self.device.type == "cpu":
            return t
        self.keep_alive.append(t)
        return t.to(self.device, non_blocking=True)

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return self.ops.apply_rope(x, cos, sin, self.rotary_dim)

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

        # Write this step's keys and values into the cache (positions self.position on), growing it by whole blocks
        end = self.position + seq_len
        cache = self.kv_cache[layer]
        if cache is None or cache[0].shape[1] < end:
            capacity = -(-end // KV_BLOCK) * KV_BLOCK
            grown = tuple(k.new_zeros(self.num_kv_heads, capacity, self.head_dim) for _ in range(2))
            if cache is not None:
                for new, old in zip(grown, cache):
                    new[:, :self.position] = old[:, :self.position]
            cache = self.kv_cache[layer] = grown
        cache[0][:, self.position:end] = k
        cache[1][:, self.position:end] = v
        k, v = cache
        out = self.ops.attention_core(q, k, v, self.attention_mask(seq_len, k.shape[1]), gate,
                                      self.num_heads // self.num_kv_heads, self.head_dim ** -0.5)
        return self.linear(out, p + ".o_proj")

    def attention_mask(self, seq_len: int, capacity: int) -> torch.Tensor:
        """[seq_len, capacity] float32, 0 where query i (at position self.position + i) may attend to key j, else
        -inf: causal, and nothing past the keys written so far. The same for every attention layer in a forward
        pass, so it's built once (on the CPU, so no GPU op is set up per position) and copied without waiting."""
        key = (self.position, seq_len, capacity)
        if self.mask_key != key:
            limit = torch.arange(self.position, self.position + seq_len)[:, None]
            mask = torch.zeros(seq_len, capacity).masked_fill(torch.arange(capacity)[None, :] > limit, float("-inf"))
            self.keep_alive.append(mask)
            self.mask, self.mask_key = mask.to(self.device, non_blocking=self.device.type != "cpu"), key
        return self.mask

    def token_mixer(self, x: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return self.attention(x, layer, cos, sin)

    def tick(self, section: str, since: float) -> float:
        """Adds the time since `since` to a section's total; returns now, so calls can be chained."""
        if self.sync_timings and self.device.type == "mps":
            torch.mps.synchronize()
        now = time.perf_counter()
        self.timings[section] += now - since
        return now

    def guess_next_layer(self, residual: torch.Tensor, layer: int):
        """Guesses the next layer's experts by running its router on the current residual stream (it hasn't
        been through this layer's experts or the next layer's attention yet, but usually picks many of the
        same experts), so moe can start reading them while this layer and the next attention run. Returns
        their ids as a tensor (on the engine's device), or None. Decode only: a prefill batch reads most
        experts anyway."""
        following = layer + 1
        if residual is None or not self.prefetch_top_k or following >= self.num_layers or residual.shape[0] != 1:
            return None
        p = f"{self.prefix}layers.{following}"
        logits = self.linear(self.rms_norm(residual, p + ".post_attention_layernorm.weight"), p + ".mlp.gate")
        top = logits[0].topk(self.prefetch_top_k)
        if self.min_expert_weight:
            self.guess_weights = F.softmax(top.values, dim=-1, dtype=torch.float32)
        return top.indices

    def skip_small_experts(self, routing_weights: torch.Tensor, selected: torch.Tensor, guess):
        """Drops routed experts below min_expert_weight, trading some accuracy for fewer reads and matmuls. Each
        token keeps its top expert, and its remaining weights are scaled back up to the same total. A skipped entry
        of selected becomes -1. Guesses for the next layer below the threshold are dropped too, so they aren't read.
        Returns (routing_weights, selected, guess)."""
        keep = routing_weights.float().cpu() >= self.min_expert_weight
        keep[:, 0] = True  # topk sorts by weight, so this is the top expert
        if self.ops.decoding:
            self.experts_routed += keep.numel()
            self.experts_skipped += keep.numel() - int(keep.sum())
        if guess is not None:
            guess = [expert for expert, weight in zip(guess, self.guess_weights.cpu().tolist())
                     if weight >= self.min_expert_weight]
        self.keep_alive.append(keep)
        kept = routing_weights * keep.to(routing_weights.device, non_blocking=routing_weights.device.type != "cpu")
        kept = kept * (routing_weights.sum(dim=-1, keepdim=True) / kept.sum(dim=-1, keepdim=True))
        return kept, selected.masked_fill(~keep, -1), guess

    def combine_decode_experts(self, outputs: dict, selected: torch.Tensor, routing_weights: torch.Tensor):
        """Weights one decode token's expert outputs and adds them up, in expert order like the general path in moe,
        for the same bf16 result. Not compiled: a fused kernel skips the rounding between these steps, which changes
        the output."""
        order = selected[0].argsort()
        order = order[selected[0, order] >= 0]  # leaves out experts skip_small_experts dropped
        self.keep_alive.append(order)
        scales = routing_weights[0, order.to(routing_weights.device, non_blocking=True), None, None]
        parts = (torch.stack([outputs[expert] for expert in sorted(outputs)]) * scales).unbind(0)
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total

    def moe(self, x: torch.Tensor, layer: int, residual: torch.Tensor = None) -> torch.Tensor:
        p = f"{self.prefix}layers.{layer}.mlp"
        t = time.perf_counter()
        routing_weights, selected = self.ops.route(self.linear(x, p + ".gate"), self.top_k, self.norm_topk_prob)
        t = self.tick("router", t)
        guess = self.guess_next_layer(residual, layer)
        t = self.tick("prefetch", t)
        # Which experts each token picked (and the guesses) are needed on the CPU: to read them, and to index
        # with (nonzero on the GPU waits for it every time), so both come back in one copy. On the GPU it's the
        # layer's one wait for all queued GPU work, so after it the GPU has run every earlier copy, and the
        # expert slots it read are free to refill.
        if guess is None:
            selected = selected.cpu()
        else:
            both = torch.cat((selected.flatten(), guess)).cpu()
            selected, guess = both[:selected.numel()].view(selected.shape), both[selected.numel():].tolist()
        t = self.tick("gpu wait", t)
        self.keep_alive = []
        if self.min_expert_weight:
            routing_weights, selected, guess = self.skip_small_experts(routing_weights, selected, guess)
        used = selected[selected >= 0]
        self.experts.record_usage(layer, used)
        # Start this layer's reads first, so the guesses for the next layer queue up behind them
        loads = self.experts.load_many(layer, used.unique().tolist())
        if guess is not None:
            self.experts.prefetch(layer + 1, guess)
        t = self.tick("read setup", t)

        # Run every token routed to each expert as it arrives (cache hits first, then reads as they finish).
        # "expert load" is only the time spent waiting for a read that hasn't finished yet.
        # An expert whose weights are all on the GPU (a GGUF pack built with --gpu) is computed there, and its
        # output stays there until the layer's sum is copied back once, so the CPU doesn't wait on every expert.
        # Copies to the GPU don't wait (a waiting copy waits for all queued GPU work: ~0.2 ms each), so their CPU
        # sources are kept in self.keep_alive until the GPU's next wait (the copy back below, or the next layer's
        # `selected.cpu()`), by which time they have all run.
        # A decode token whose hidden state and fused expert are all on the GPU skips the per-expert indexing (the
        # one token is every expert's input), which issues far fewer GPU ops. See combine_decode_experts.
        results, on_device = [], {x.device: (x, routing_weights)}
        decode_outputs = {}  # expert -> its output before the routing weight
        fast_decode = self.ops.compiled and self.ops.decoding
        for expert, weights in loads:
            t = self.tick("expert load", t)
            if fast_decode and len(weights) == 2 and all(w.device.type == x.device.type for w in weights):
                decode_outputs[expert] = F.linear(self.ops.expert_gate(F.linear(x, weights[0])), weights[1])
                t = self.tick("expert compute", t)
                continue
            token_idx, slot = (selected == expert).nonzero(as_tuple=True)
            device = weights[0].device if all(w.device == weights[0].device for w in weights) else x.device
            if device not in on_device:  # this layer's input and routing weights, copied there once
                gpu = device.type != "cpu"
                on_device[device] = (x.to(device, non_blocking=gpu), routing_weights.to(device, non_blocking=gpu))
                self.keep_alive.append(on_device[x.device])
            if device.type != "cpu":
                index = torch.stack((token_idx, slot))
                self.keep_alive.append(index)
                token_idx, slot = index.to(device, non_blocking=True)
            x_dev, weights_dev = on_device[device]
            h, scale = x_dev[token_idx], weights_dev[token_idx, slot, None]
            if len(weights) == 2:  # fused gate_up: one matmul, then split its output
                gate, up = F.linear(h, weights[0]).chunk(2, dim=-1)
            else:
                gate, up = F.linear(h, weights[0]), F.linear(h, weights[1])
            h = F.linear(F.silu(gate) * up, weights[-1])
            results.append((expert, token_idx, h * scale))
            t = self.tick("expert compute", t)
        if decode_outputs and not results:
            out = self.combine_decode_experts(decode_outputs, selected, routing_weights)
        else:
            for expert, h in decode_outputs.items():  # a layer that also has an expert the fast path can't run
                token_idx, slot = (idx.to(x.device) for idx in (selected == expert).nonzero(as_tuple=True))
                results.append((expert, token_idx, h * routing_weights[token_idx, slot, None]))
            # Experts arrive in whatever order their reads finish; adding them up in a fixed order keeps the
            # (bf16) output, and so greedy decoding, the same from run to run
            sums = {}
            for _, token_idx, h in sorted(results, key=lambda r: r[0]):
                if h.device not in sums:
                    sums[h.device] = torch.zeros_like(x, device=h.device)
                sums[h.device].index_add_(0, token_idx, h)
            out = torch.zeros_like(x)
            for total in sums.values():
                out += total.to(x.device)
        self.tick("expert compute", t)

        sp = p + ".shared_expert"
        if sp + ".gate_proj.weight" in self.weights:
            t = time.perf_counter()
            out = out + self.ops.shared_expert(x, self.w(sp + ".gate_proj.weight"), self.w(sp + ".up_proj.weight"),
                                               self.w(sp + ".down_proj.weight"), self.w(p + ".shared_expert_gate.weight"))
            self.tick("shared expert", t)
        return out

    @torch.no_grad()
    def forward(self, token_ids: list) -> torch.Tensor:
        """Runs new tokens through the model, extending the caches. Returns logits for each token."""
        self.ops.decoding = len(token_ids) == 1
        table, ids = self.w(self.prefix + "embed_tokens.weight"), torch.tensor(token_ids)
        # A GGUF table kept quantized (gguf.quants.QuantizedEmbedding) looks rows up on the CPU
        x = table[self.to_device(ids)] if isinstance(table, torch.Tensor) else self.to_device(table[ids])
        cos, sin = self.rope(len(token_ids))
        for layer in range(self.num_layers):
            p = f"{self.prefix}layers.{layer}"
            t = time.perf_counter()
            x = x + self.token_mixer(self.rms_norm(x, p + ".input_layernorm.weight"), layer, cos, sin)
            self.tick(self.layer_types[layer].replace("_", " "), t)
            x = x + self.moe(self.rms_norm(x, p + ".post_attention_layernorm.weight"), layer, residual=x)
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
        return time.perf_counter(), e.bytes_read, e.read_seconds, e.hits, e.reads, e.prefetches, e.prefetch_used

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
            "prefetches": e.prefetches - start[5],
            "prefetch_used": e.prefetch_used - start[6],
            "timings": timings,
        }

    def generate(self, prompt: str, max_new_tokens: int = 100, stream: bool = False) -> list:
        """Greedy decoding from a fresh start. Returns prompt + generated token ids; timings land in self.stats."""
        self.reset()
        return self.continue_with(self.prompt_ids(prompt), max_new_tokens, stream)

    def continue_with(self, ids: list, max_new_tokens: int = 100, stream: bool = False, on_text=None) -> list:
        """Runs ids through the model after everything it has already seen (the caches are kept, so a chat only
        processes each new message), then decodes greedily. Returns ids + generated token ids; timings land in
        self.stats. Stopping on an end token doesn't run that token through the model.
        on_text, if given, is called with each new piece of the reply's text; it returns True to stop the reply
        there, after that piece's token has run through the model, so the caches stay whole and a chat can go on."""
        self.stats = {}
        ids = list(ids)
        num_prompt = len(ids)
        printed = ""
        start = self.io_snapshot()
        logits = self.forward(ids)
        self.record_phase("prefill", num_prompt, start)

        # Decode phase: each step forwards the previous token. The first token comes from the prefill logits,
        # so the timed decode tokens are the forward passes after it.
        start = self.io_snapshot()
        self.experts_routed = self.experts_skipped = 0
        decode_steps = 0
        for _ in range(max_new_tokens):
            next_id = int(logits[-1].argmax())
            if next_id in self.eos_ids:
                break
            ids.append(next_id)
            stop = False
            if stream or on_text:
                # Decode the whole reply so multi-byte characters split across tokens come out correctly
                text = self.tokenizer.decode(ids[num_prompt:])
                piece, printed = text[len(printed):], text
                if stream:
                    print(piece, end="", flush=True)
                if on_text:
                    stop = bool(on_text(piece))
            logits = self.forward([next_id])
            decode_steps += 1
            if stop:
                break
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
        if self.ops.compiled and self.ops.decoding:
            # A decode token on the GPU: the rest of the layer as one compiled function
            out, self.conv_state[layer], self.recurrent_state[layer] = self.ops.linear_attention_step(
                mixed, z, b, a, self.conv_state[layer], self.w(p + ".conv1d.weight"), self.w(p + ".A_log"),
                self.w(p + ".dt_bias"), self.w(p + ".norm.weight"), self.recurrent_state[layer],
                self.lin_k_heads, self.lin_v_heads, self.lin_k_dim, self.lin_v_dim, self.eps)
            return self.linear(out, p + ".out_proj")

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
            state = torch.zeros(self.lin_v_heads, self.lin_k_dim, self.lin_v_dim, device=q.device)
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
        # Qwen chat format; an empty think block turns thinking off. Mid-chat (position > 0), it first closes the
        # previous reply: continue_with stops on its end token without running it through the model.
        text = "<|im_end|>\n" if self.position else ""
        text += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        text += "<think>\n" if self.think else "<think>\n\n</think>\n\n"
        return self.tokenizer.encode(text, add_special_tokens=False).ids

ENGINES = {"qwen3_moe": MoEEngine, "qwen3_5_moe": Qwen35MoEEngine}

def load_engine(model_dir: str, pack_dir: str = None, cache_gb: float = None) -> MoEEngine:
    with open(os.path.join(model_dir, "config.json")) as f:
        full_cfg = json.load(f)
    model_type = full_cfg.get("model_type")
    if model_type not in ENGINES:
        raise NotImplementedError(f"model_type {model_type!r} is not supported (supported: {', '.join(ENGINES)})")
    engine_class = ENGINES[model_type]
    cfg = full_cfg.get("text_config", full_cfg)

    eos = cfg.get("eos_token_id")
    gen_cfg_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        with open(gen_cfg_path) as f:
            eos = json.load(f).get("eos_token_id", eos)
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    # Everything except the routed experts stays resident in RAM. get_tensor only memory-maps the shard,
    # so clone() to really load it: otherwise macOS drops the pages under memory pressure and silently
    # re-reads them from disk on every token. The one exception is the embedding table when lm_head is
    # separate: each token reads a single row of it, so leaving it memory-mapped saves ~1 GB for free.
    index_map = build_index_map(model_dir)
    lazy = {engine_class.prefix + "embed_tokens.weight"} if "lm_head.weight" in index_map else set()
    weights = {}
    for shard in sorted(set(index_map.values())):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if not EXPERT_RE.match(name) and not name.startswith(SKIP_PREFIXES):
                    tensor = f.get_tensor(name)
                    weights[name] = tensor if name in lazy else tensor.clone()

    # No cache by default: bf16 experts are big, and the OS file cache holds recent ones without the memory pressure
    experts = ExpertStore(pack_dir or os.path.join(model_dir, "expert_pack"), int((cache_gb or 0) * 1024**3))
    tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    return engine_class(cfg, eos_ids, weights, experts, tokenizer)

def run_chat(model_dir: str, max_new_tokens: int = 2048, pack_dir: str = None, think: bool = False,
             cache_gb: float = None, load=load_engine):
    """An interactive chat (`main.py chat`). The model keeps the whole conversation in its caches, so each reply
    only processes the new message. /reset starts a new conversation; /exit, Ctrl-D or Ctrl-C quits."""
    engine = load(model_dir, pack_dir, cache_gb)
    if not isinstance(engine, Qwen35MoEEngine):
        raise SystemExit("chat needs a Qwen3.5/3.6 model (it uses their chat format)")
    engine.think = think
    print("\nchat: type a message and press Enter. /reset starts over, /exit (or Ctrl-D) quits.")
    try:
        while True:
            try:
                message = input("\n> ").strip()
            except EOFError:
                break
            if message == "/exit":
                break
            if message == "/reset":
                engine.reset()
                print("(new conversation)")
                continue
            if not message:
                continue
            print()
            engine.continue_with(engine.prompt_ids(message), max_new_tokens, stream=True)
            read, reply = engine.stats["prefill"], engine.stats["decode"]
            speed = f" at {reply['tokens'] / reply['seconds']:.1f} tok/s" if reply["tokens"] else ""
            print(f"\n[{reply['tokens']} tokens{speed}; read your message in {read['seconds']:.1f}s; "
                  f"conversation so far: {engine.position} tokens]")
    except KeyboardInterrupt:  # mid-reply, this leaves the caches half-updated, so it quits rather than continues
        print()
    finally:
        engine.experts.save_usage()

def run_prompt(model_dir: str, prompt: str, max_new_tokens: int = 2048, pack_dir: str = None,
               think: bool = False, cache_gb: float = None, load=load_engine):
    """Generates a reply to prompt, streaming it to stdout, then prints speed and I/O stats (`main.py run`).
    load builds the engine from model_dir: load_engine here, gguf.engine.load_engine for .gguf files."""
    start = time.perf_counter()
    engine = load(model_dir, pack_dir, cache_gb)
    load_seconds = time.perf_counter() - start
    engine.think = think
    if isinstance(engine, Qwen35MoEEngine):
        print(f"> {prompt}\n")
    else:
        print(prompt, end="", flush=True)
    engine.generate(prompt, max_new_tokens, stream=True)
    engine.experts.save_usage()

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
        if s["prefetches"]:
            print(f"{'':9} prefetch: {s['prefetches']} next-layer reads started, "
                  f"{s['prefetch_used'] / s['prefetches'] * 100:.0f}% of them used")
    cache = engine.experts
    print(f"cache:    {cache.cached_bytes / 1024**3:.2f} / {cache.cache_bytes / 1024**3:.2f} GiB used, "
          f"{len(cache.cache)} experts held")
    if engine.experts_routed:
        print(f"skipped:  {engine.experts_skipped} of {engine.experts_routed} routed experts while decoding "
              f"({engine.experts_skipped / engine.experts_routed * 100:.0f}%), weight below {engine.min_expert_weight}")

    decode = engine.stats.get("decode")
    if decode and decode["tokens"]:
        print("\ndecode time per token, by section:")
        for section, seconds in sorted(decode["timings"].items(), key=lambda kv: -kv[1]):
            ms = seconds / decode["tokens"] * 1000
            print(f"  {section:18s} {ms:8.1f} ms  ({seconds / decode['seconds'] * 100:4.1f}%)")
