"""Qwen3.8-Flash-Next (GGUF architecture qwen4exp; transformers' qwen4_exp). Qwen3.6's layers (Gated DeltaNet,
gated full attention, a shared expert plus 10 of 512 routed experts), wired differently:

- The residual stream is hc_count (4) parallel streams, [tokens, 4 * hidden]. Every attention or MoE block reads one
  hidden-wide input from them through a gated mix (hc_read), and adds its output back to each stream, scaled by a
  per-stream gate (hc_write). They take the place of the layer norms; a last mix feeds lm_head.
- One layer first adds hashed bigram/trigram embeddings (PLE) to the streams; the 27 GiB table stays on disk.
- The Gated DeltaNet's output norm is gated with a sigmoid, not silu.
- Full attention is dense. Qwen's QSA indexer would only drop tokens once the context passes dense_limit (2,051
  tokens), so until then this is exact; past it the output starts to differ from the reference.

Written against transformers' modular_qwen4_exp.py (Qwen4ExpTextGatedResidual, Qwen4ExpTextNGramEmbedding,
Qwen4ExpTextPLELayer, Qwen4ExpTextDecoderLayer, Qwen4ExpTextModel)."""
import math
import os
import time

import torch
import torch.nn.functional as F

from engines.base import Engine, GatedDeltaNet, QwenChat

def grouped_rms_norm(x: torch.Tensor, weight: torch.Tensor, groups: int, eps: float) -> torch.Tensor:
    """Zero-centered RMSNorm over each of groups equal slices of the last dim (one per stream), then scaled by
    (1 + weight) over the whole of it."""
    x32 = x.float().reshape(*x.shape[:-1], groups, -1)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32.reshape(x.shape) * (1.0 + weight.float())).type_as(x)

def hc_matmul(x: torch.Tensor, weight) -> torch.Tensor:
    """F.linear(x, weight), for a bf16 weight or an int8 one given as (q, scales): the same math as
    quants.Int8Weight.linear, written on its tensors so torch.compile can trace it."""
    if isinstance(weight, torch.Tensor):
        return F.linear(x, weight)
    q, scales = weight
    y = torch._weight_int8pack_mm(x.reshape(-1, x.shape[-1]).to(scales.dtype), q, scales)
    return y.view(*x.shape[:-1], y.shape[-1]).to(x.dtype)

def hc_step(x, out, gates, norm_w, down, up, inject, hc: int, eps: float):
    """The gated residual (Qwen4ExpTextGatedResidual), as a plain function of tensors so it can be compiled.
    If out is given, first adds that block's output [tokens, hidden] to every stream of x [tokens, hc * hidden],
    scaled by the stream's gate from gates [tokens, hc] (the write). Then reads the next block's input from the
    streams (the read). -> (x, the block's input [tokens, hidden], its write gates [tokens, hc], or None when
    inject is None)."""
    if out is not None:
        x = x + (out[:, None, :] * gates[:, :, None]).flatten(1)
    xn = grouped_rms_norm(x, norm_w, hc, eps)
    read = F.silu(hc_matmul(xn, down) / hc)
    read = torch.sigmoid(hc_matmul(read, up))
    shape = (x.shape[0], hc, x.shape[1] // hc)
    mixed = (read.view(shape) * xn.view(shape)).mean(dim=-2)
    if inject is None:
        return x, mixed, None
    return x, mixed, 2 * torch.sigmoid(hc_matmul(xn, inject) / hc)

class Qwen38Engine(GatedDeltaNet, QwenChat, Engine):
    prefetch_top_k = 10  # Qwen3.8 routes to 10 experts
    prefix = "model.language_model."
    zero_centered_norm = True
    gdn_output_gate = "sigmoid"

    def __init__(self, cfg: dict, eos_ids: set, weights: dict, experts, tokenizer, device: str = "cpu",
                 ple_table=None):
        """ple_table: the n-gram table, anything that table[row ids] looks rows up in (gguf.streamed.StreamedEmbedding
        for a GGUF). Needed when the config has PLE layers."""
        self.ple_table = ple_table
        super().__init__(cfg, eos_ids, weights, experts, tokenizer, device)
        # While decoding on the GPU, hc_step is compiled: it's ~15 small ops, run 3 times a layer (attention, MoE,
        # and the next layer's expert guess). emulate_precision_casts rounds to bf16 wherever eager torch does,
        # so the compiled steps give exactly the eager result. WILLOW_COMPILED_HC=0 runs it as it is
        self.hc_step = hc_step
        if self.ops.compiled and os.environ.get("WILLOW_COMPILED_HC", "1") != "0":
            compiled = torch.compile(hc_step, dynamic=False, options={"emulate_precision_casts": True})
            self.hc_step = self.ops.dispatch(hc_step, compiled)
        self.hc_args = {}  # hyper-connection prefix -> its weights, as hc_step takes them

    def setup(self, cfg: dict):
        super().setup(cfg)
        self.norm_topk_prob = cfg.get("norm_topk_prob", True)
        self.setup_linear_attention(cfg)
        self.hidden = cfg["hidden_size"]
        self.hc_count = cfg["hc_count"]
        self.ple_layers = set(cfg.get("ple_layers") or [])
        if self.ple_layers:
            if self.ple_table is None:
                raise ValueError("this model has PLE layers, so it needs ple_table")
            self.ngram_size = cfg["ngram_size"]
            self.heads_per_ngram = cfg["heads_per_ngram"]
            self.ple_eos = cfg["ple_eos_token_id"]
            self.ple_multipliers = torch.tensor(cfg["ple_layer_multipliers"], dtype=torch.int64)
            self.ple_head_sizes = torch.tensor(cfg["ple_head_vocab_sizes"], dtype=torch.int64)
            self.ple_head_offsets = torch.tensor(cfg["ple_head_offsets"], dtype=torch.int64)
            self.ple_kernel = cfg["ple_conv_kernel_size"]
            self.ple_dilation = self.ngram_size  # as in the reference: the conv's dilation is the n-gram size
        budget, ratio = cfg.get("indexer_budget"), cfg.get("indexer_compress_ratio")
        # QSA keeps the top budget / ratio blocks of ratio tokens plus the incomplete tail, so a query that sees at
        # most (budget / ratio + 1) * ratio - 1 tokens keeps all of them
        self.dense_limit = (budget // ratio + 1) * ratio - 1 if budget and ratio else None

    def reset(self):
        super().reset()
        self.reset_linear_attention()
        if self.ple_layers:
            self.ple_history = [self.ple_eos] * (self.ngram_size - 1)  # the last tokens, for the next call's n-grams
            self.ple_conv_state = None  # the last (kernel - 1) * dilation inputs of the PLE conv
        self.warned_dense = False

    def token_mixer(self, x: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.layer_types[layer] == "linear_attention":
            return self.linear_attention(x, layer)
        return self.attention(x, layer, cos, sin)

    # The gated residual (Qwen4ExpTextGatedResidual)

    def hc(self, x: torch.Tensor, p: str, pending: tuple = None, write: bool = True):
        """Adds pending, a block's (output, write gates), to the streams x [tokens, hc * hidden] if given, then reads
        the next block's input through the mixer at p (see hc_step). -> (x, the block's input [tokens, hidden], its
        write gates [tokens, hc] or None)."""
        if p not in self.hc_args:
            def weight(name):
                assert self.weights.get(f"{p}.{name}.bias") is None
                w = self.weights.get(f"{p}.{name}.weight")
                return w if w is None or isinstance(w, torch.Tensor) else (w.q, w.scales)  # an Int8Weight
            self.hc_args[p] = (self.w(p + ".hc_norm.weight"), weight("input_mix_weight_down"),
                               weight("input_mix_weight_up"), weight("block_inject_weight"))
        norm_w, down, up, inject = self.hc_args[p]
        out, gates = pending if pending is not None else (None, None)
        return self.hc_step(x, out, gates, norm_w, down, up, inject if write else None, self.hc_count, self.eps)

    @staticmethod
    def hc_write(x: torch.Tensor, out: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """Adds a block's output [tokens, hidden] to every stream, scaled by that stream's gate."""
        return x + (out[:, None, :] * gates[:, :, None]).flatten(1)

    def router_guess_input(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        return self.hc(residual, f"{self.prefix}layers.{layer}.mlp_hyper_connection", write=False)[1]

    # PLE: hashed n-gram embeddings (Qwen4ExpTextNGramEmbedding and Qwen4ExpTextPLELayer)

    def ple_rows(self, token_ids: list) -> torch.Tensor:
        """The n-gram table rows of each new token [tokens, (ngram_size - 1) * heads_per_ngram]: for n = 2..ngram_size,
        the token and the n - 1 before it, hashed into each of heads_per_ngram prime-sized buckets. Tokens before
        the start, or before an EOS, count as EOS."""
        n, eos = self.ngram_size, self.ple_eos
        history = torch.tensor(self.ple_history + list(token_ids), dtype=torch.int64)
        self.ple_history = history[-(n - 1):].tolist()
        pos = torch.arange(len(history))
        # Where each token's segment starts: just after the last EOS before it
        last_eos = torch.cummax(torch.where(history == eos, pos, -1), dim=0).values
        segment_start = torch.cat((torch.tensor([-1]), last_eos[:-1])) + 1
        shifted = []
        for shift in range(n):
            source = pos - shift
            valid = (pos - segment_start >= shift) & (source >= 0)
            shifted.append(torch.where(valid, history[source.clamp_min(0)], eos))
        rows = []
        for size in range(2, n + 1):
            mixed = shifted[0] * self.ple_multipliers[0]
            for i in range(1, size):
                mixed = mixed ^ (shifted[i] * self.ple_multipliers[i])
            heads = slice((size - 2) * self.heads_per_ngram, (size - 1) * self.heads_per_ngram)
            rows.append(torch.remainder(mixed[:, None], self.ple_head_sizes[None, heads]) + self.ple_head_offsets[None, heads])
        return torch.cat(rows, dim=-1)[-len(token_ids):]

    def ple(self, x: torch.Tensor, token_ids: list, layer: int) -> torch.Tensor:
        """What PLE adds to the streams [tokens, hc * hidden]."""
        p = f"{self.prefix}layers.{layer}.ple"
        hc, tokens = self.hc_count, x.shape[0]
        rows = self.ple_rows(token_ids)
        embeddings = self.to_device(self.ple_table[rows].reshape(tokens, -1)).to(x.dtype)
        key = grouped_rms_norm(self.linear(embeddings, p + ".key_proj"), self.w(p + ".norm_key.weight"), hc, self.eps)
        query = grouped_rms_norm(x, self.w(p + ".norm_query.weight"), hc, self.eps)
        value = self.linear(embeddings, p + ".value_proj")
        shape = (tokens, hc, self.hidden)
        gate = (key.view(shape) * query.view(shape)).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value[:, None, :]).flatten(1)
        return gated + self.ple_conv(grouped_rms_norm(gated, self.w(p + ".norm_conv.weight"), hc, self.eps), p)

    def ple_conv(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """silu of a causal depthwise conv with dilation, continuing from the previous calls' inputs."""
        kernel, dilation = self.ple_kernel, self.ple_dilation
        history = (kernel - 1) * dilation
        state = self.ple_conv_state if self.ple_conv_state is not None else x.new_zeros(history, x.shape[1])
        window = torch.cat((state, x))
        self.ple_conv_state = window[-history:]
        weight = self.w(p + ".conv1d.weight").float()  # [channels, kernel]
        window, tokens = window.float(), x.shape[0]
        conv = sum(window[k * dilation:k * dilation + tokens] * weight[:, k] for k in range(kernel))
        return F.silu(conv.to(x.dtype))

    @torch.no_grad()
    def forward(self, token_ids: list) -> torch.Tensor:
        self.ops.decoding = len(token_ids) == 1
        if self.dense_limit and self.position + len(token_ids) > self.dense_limit and not self.warned_dense:
            print(f"\nnote: past {self.dense_limit} tokens, Qwen3.8's sparse attention (QSA) would leave some tokens "
                  "out; willow still attends to all of them, so its output starts to differ from the reference")
            self.warned_dense = True
        x = self.embed(token_ids).repeat(1, self.hc_count)  # hc identical copies of the embedding
        cos, sin = self.rope(len(token_ids))
        # Each block's output is added to the streams by the next hc call (one compiled step for the write and the
        # read that follows it), so pending is (the last block's output, its write gates) until then
        pending = None
        for layer in range(self.num_layers):
            p = f"{self.prefix}layers.{layer}"
            t = time.perf_counter()
            if layer in self.ple_layers:
                if pending is not None:  # PLE reads the streams
                    x, pending = self.hc_write(x, *pending), None
                x = x + self.ple(x, token_ids, layer)
                t = self.tick("ple", t)
            x, h, gates = self.hc(x, p + ".attn_hyper_connection", pending)
            t = self.tick("hyper-connections", t)
            out = self.token_mixer(h, layer, cos, sin)
            t = self.tick(self.layer_types[layer].replace("_", " "), t)
            x, h, gates = self.hc(x, p + ".mlp_hyper_connection", (out, gates))
            self.tick("hyper-connections", t)
            pending = (self.moe(h, layer, residual=x), gates)
        self.position += len(token_ids)
        t = time.perf_counter()
        x, h, _ = self.hc(x, self.prefix + "hyper_connection_mixer", pending, write=False)
        # Only the last token's logits are used (a prompt's others would be [tokens, vocab], GiBs for a long one)
        logits = F.linear(h[-1:], self.w("lm_head.weight"))
        self.tick("lm_head", t)
        return logits
