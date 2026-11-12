# Roadmap: Qwen3.8-Flash-Next

Goal: run **Qwen3.8-Flash-Next** (`Qwen3.8-Flash-Next-UD-IQ4_XS`, Unsloth's GGUF, 4 split files, about 87 GiB) on the same 16 GB Mac, streaming its experts from the SSD the way willow runs Qwen3.6. Qwen3.6 must keep working the whole way through.

## What the new model is

The GGUF's architecture is `qwen4exp` ("A Preview of the Qwen4 Architecture"); the original is [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`model_type: qwen4_exp`), quantized by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). The table comes from the GGUF header and the reference code (see [References](#references)); nothing has been run yet.

| | Qwen3.6-35B-A3B (today) | Qwen3.8-Flash-Next |
|---|---|---|
| Architecture | `qwen35moe` | `qwen4exp` |
| Layers | 40 | 48 |
| Hidden size | 2048 | 2560 |
| Experts | 256, 8 per token | 512, **10** per token, expert FFN 640 |
| Shared expert | yes, sigmoid-gated | yes, sigmoid-gated (FFN 640) |
| Token mixers | Gated DeltaNet ×3, then full attention | the same pattern: 36 linear layers, 12 full (every 4th) |
| Full attention | gated, GQA | gated, 24 heads / 2 KV heads, head dim 256, plus **Qwen Sparse Attention (QSA)**: an indexer (4 query heads, 1 key head, dim 128) picks the top 512 blocks of 4 tokens plus the incomplete tail. The KV cache itself isn't compressed |
| Residual stream | one 2048-wide stream | **gated residual (hyper-connections)**: 4 streams (10240 wide). Each sublayer reads them through an element-wise low-rank (320) sigmoid gate, averaged, and writes its output back to each stream with a scalar gate |
| Norms | `attn_norm`, `post_attention_norm`, `output_norm` | none of these: `hc_attn_norm`, `hc_ffn_norm`, `output_hc_norm` instead |
| Extra embeddings | none | **n-gram embedding (PLE)** before layer index 1 (the model card's "layer 2"): bigrams and trigrams hashed into 16 heads of prime-sized buckets, 320M rows × 160 dims, gated per stream and added to all 4 |
| Quant types | Q4_K, Q6_K | IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_S, Q5_0, Q6_K, Q8_0, BF16, F32 and **Q2_0** (ggml type 42, new in llama.cpp; 2.25 bits/weight) |
| Linear attention | Gated DeltaNet, SiLU-gated output norm | the same, but the output norm's gate is a **sigmoid** |
| Tokenizer and chat format | `gpt2` BPE, `qwen35` pre-tokenizer, `<think>` | the same |
| Sampling defaults | (greedy in willow) | thinking: temp 1.0, top-p 0.95, top-k 20; no thinking: temp 0.7, top-p 0.8, top-k 20, presence penalty 1.5 |

Sizes on disk: experts 56.2 GiB, the PLE table 26.8 GiB, everything else 4.2 GiB.

## What this means for willow

- **The GGUF reader reads the file now** (Q2_0 added), but the IQ types still have no dequantizers.
- **Per-token I/O roughly doubles.** That's 480 experts per token (48 × 10), against 320 for Qwen3.6. An expert is about 2.3 MiB as stored, against 1.9 MiB, so a token that misses every cache reads about 1.1 GiB (Qwen3.6: about 0.6 GiB). The usage table, preload and next-layer prefetch matter even more here.
- **The PLE n-gram table stays out of RAM** (26.8 GiB). It is the only tensor in shard 4 (a 192-byte header, then 320,001,536 rows × 90 bytes of IQ4_NL), so it is read in place, row by row, with no copy and no extra disk space (`gguf/streamed.py`). Each token needs 16 rows: about 1.5 ms per decode token, and 0.16 s for a 512-token prompt's 8,192 random rows, small next to the experts.
- **RAM is tight.** Dequantized to bf16 like Qwen3.6's, the non-expert weights take about 8 GiB (plus 1.2 GiB for the embedding table, if it can't stay quantized). Some weights may need to stay 8-bit.
- **The forward pass is new.** Hyper-connections change how every layer reads and writes the residual stream, so `MoEEngine.forward` can't be reused as is. The Gated DeltaNet layers, the expert streaming (`moe`), the KV cache, the GPU paths and the generate loop are all worth sharing.
- **There is a reference implementation in Python.** transformers' `modular_qwen4_exp.py` is written as a diff over Qwen3.5, which is what willow's engine ports, so the new parts map almost line by line. llama.cpp's `qwen4exp.cpp` is a second reference, and can give reference logits for this exact GGUF.
- **Short contexts need no sparse attention.** Until there are more than 512 complete blocks of 4 visible tokens (about 2,048 tokens), the indexer keeps every token, so dense attention is exact.

## Plan

### 0. Isolate Qwen3.6, remove safetensors

- [x] Remove the safetensors path: its loader, its pack compiler and its reference checks, plus `requirements-ref.txt` and the `safetensors`/`transformers` dependencies.
- [x] Split the code into `gguf/` (format: reader, quants, pack, expert store, tokenizer), `engines/qwen36/` (Qwen3.6 model and loader) and `runner.py` (CLI loops). `engines/__init__.py` picks an engine from `general.architecture`.
- [x] Confirm Qwen3.6 gives the same output as before the split (same greedy reply, same speed).

### 1. Find the reference

- [x] The llama.cpp source: `src/models/qwen4exp.cpp` on master ([`09338b8`](https://github.com/ggml-org/llama.cpp/tree/09338b85e8b230742fa56f93a1d9f0bf006ca2d3)); type 42 is `GGML_TYPE_Q2_0`.
- [x] Hugging Face modeling code: `transformers/models/qwen4_exp/` on main ([`f210a32`](https://github.com/huggingface/transformers/tree/f210a32fbf2e3213a143dad8c587935216088b38/src/transformers/models/qwen4_exp)).
- [x] The math for the gated residual, PLE, QSA and the GDN difference: see [How Qwen3.8 differs](#how-qwen38-differs-from-the-reference-code).
- [ ] Reference logits: build llama.cpp at that commit and dump the logits for a short prompt on this GGUF (e.g. `llama-eval-callback` or a tiny script with `llama_get_logits`).

### 2. Format support (`gguf/`)

- [x] Reader: type 42 is Q2_0 (64-value blocks: an fp16 scale, then 2 bits per value, mapping 0..3 to -1..+2).
- [x] Q2_0 dequantizer, checked exactly against ggml's `dequantize_row_q2_0` on layer 8's experts.
- [x] Expert pack, first version: experts copied out as their raw ggml bytes (IQ3_XXS, IQ4_XS, IQ2_S gate_up; Q5_0, IQ4_NL, type 42 down), gate and up fused. Each layer's slots are sized to its own largest expert (1.45–2.73 MiB), giving 56.4 GiB instead of 65.6 GiB padded to one size. It will be rebuilt once IQ experts are converted at pack time (see below).
- [x] IQ4_NL dequantizer (the PLE table, and the `ffn_down_exps` of 13 layers).
- [x] `StreamedEmbedding`: looks up rows of a GGUF tensor straight from the file (`pread` per row, on a thread pool for big batches), never resident. Used for the PLE table.
- [x] Dequantizers for IQ4_XS, IQ3_XXS and IQ2_S in `quants.py`, with ggml's grids in `gguf/iq_grids.py`. All of them, IQ4_NL included, match gguf-py exactly on real blocks of this model's tensors, so every type in the GGUF now dequantizes.
- [x] `QuantizedEmbedding` for IQ4_XS, so `token_embd` stays quantized (it works for any type with a dequantizer).
- [x] A pack layout for IQ experts: (a) requantizing to INT4_G32 (or INT4_G32_GPU) at pack time, so they run on the int4 kernels. The other options were (b) keeping the IQ bytes and dequantizing on every read (slow), and (c) a Metal kernel that runs IQ blocks directly (the most work). Q5_0, IQ4_NL, IQ4_XS, IQ3_XXS, IQ2_S and Q2_0 are dequantized, then requantized. For each group of 32 values, a search picks whichever range (min..max, shrunk by up to 20%, or the values' own even grid) gives the least error. Against the GGUF's weights this adds 0.7% (Q2_0), 4.3% (IQ3_XXS), 6.8% (IQ2_S) and 7.2–7.6% (IQ4_XS, IQ4_NL, Q5_0) relative error; plain min..max would give 2.4–8.4%. Every expert becomes 2.93 MiB, so the pack is about 70.5 GiB (up from 56.4), and a cold token reads about 1.4 GiB.
- [ ] If the 7% shows in the output: an exact GPU path for the 4-bit IQ types (a 16-entry lookup table times the block scale, then a bf16 matmul), or keeping Q5_0 at 8 bits.
- [ ] A GGUF-only regression check that replaces the deleted reference checks: a short prompt's logits and greedy reply saved as a golden file per model.

### 3. The `qwen4exp` engine (`engines/qwen38/`)

- [x] Shared base, `engines/base.py`. `Engine` holds the ops, RoPE, attention and KV cache, `moe` streaming with its next-layer guess (through a `router_guess_input` hook) and generation. The `GatedDeltaNet` mixin has `gdn_output_gate` (`"silu"` or `"sigmoid"`), and `QwenChat` holds the chat format. Qwen3.6 keeps only its forward pass. Old and new code give identical logits on tiny random Qwen3-MoE and Qwen3.5 models, on the CPU and the GPU.
- [x] Confirmed on the real Qwen3.6: the same greedy reply as before the split.
- [x] Loader (`engines/qwen38/loader.py`): the config from `qwen4exp.*` (PLE layers are 0-based in the GGUF), and the tensor names (the hyper-connections, PLE, indexer and final mixer). It reuses Qwen3.6's converter fixes: the `-1` on zero-centered norms (now also the PLE `norm_key`/`norm_query`/`norm_conv`), `A_log`, and the V-head reorder. On the real header, all 1,079 non-expert tensors map to transformers' names. The only tensors skipped are the experts (they come from the pack) and `per_layer_token_embd.weight` (it's streamed).
- [x] The gated residual: 4 streams, a read mixer and a write gate per sublayer, and the final mixer in place of `output_norm`. The next-layer expert guess reads through the next layer's MLP mixer.
- [x] Gated full attention, dense: exact up to 2,051 tokens. Past that, willow prints a one-time note. The QSA indexer is in phase 4.
- [x] Gated DeltaNet with a sigmoid-gated output norm (`gdn_output_gate = "sigmoid"`).
- [x] PLE: the n-gram hash (token ids → 16 row ids, reset at EOS), looked up through `StreamedEmbedding` on shard 4, plus the gate and the dilated conv. The per-chat state (the last 2 tokens and 9 conv inputs) carries across calls.
- [x] Routing: softmax, then top-k, then renormalized (`norm_topk_prob`). There is no router bias or scaling.
- [x] Chat: the same format as Qwen3.6 (`QwenChat`), so `chat` and `web` use it unchanged. Not yet tried on the real model.
- [x] Matched against transformers' `Qwen4ExpForCausalLM` on a tiny random model with every Qwen3.8 part: 4 streams, a PLE layer, an EOS mid-prompt, GDN and full attention, and 8 experts top-2. The largest logit difference is 2e-6 in float32, on the CPU and on MPS, for one prefill, chunked prefills and token-by-token decoding. The test fails if the PLE, its conv, the EOS reset or the sigmoid gate is broken.
- [ ] On the real GGUF: a first run, then a match against llama.cpp's logits (step 1).

### 4. Make it fast on 16 GB

- [x] Non-expert weights in int8 (`Int8Weight`, one scale per row, torch's int8 kernel): 4.4 GiB instead of 9.2 GiB, and faster than bf16 on the CPU and the GPU on this Mac. They add about 1–2% relative error per matmul (bf16 adds 0.2%). The routers, the one-row gates, the conv weights and matrices with a dimension not divisible by 32 stay bf16. This is the default for Qwen3.8; `--backbone bf16` turns it off, and Qwen3.6 stays bf16 unless given `--backbone int8`. The first real run (bf16 on the CPU, 9.2 GiB on a 16 GB Mac) took 9.6 s per token, mostly in compute that should take about 0.1 s, so it was very likely swapping.
- [ ] Expert cache size: what `--cache-gb` fits next to the 4.4 GiB backbone on 16 GB.
- [ ] GPU paths: `torch.compile` for the hyper-connection and PLE steps, and a shared-memory slab for the 512-expert layers.
- [ ] Tune prefetch and preload for top-10 of 512 (the usage table probably matters more with this many experts).
- [ ] The QSA indexer, for contexts past about 2,048 tokens.
- [ ] Optional: sampling (temp, top-k, top-p) using the GGUF's defaults, alongside greedy.

## How Qwen3.8 differs from the reference code

From `modular_qwen4_exp.py` (each class there extends its Qwen3.5 counterpart). `x` is the 4-stream residual `[tokens, 4, 2560]`; every RMSNorm here scales by `(1 + w)`, as Qwen3.5's does, and llama.cpp's converter stores that `1 + w` (so the loader undoes it, as `fix_qwen35_tensor` does today).

**Gated residual**, once around attention and once around the MoE (`hc_attn_*`, `hc_ffn_*`):

```
xn     = rms_norm(x) per stream, scaled by hc_norm   # hc_norm: [10240]
read   = sigmoid(up(silu(down(xn) / 4)))             # down: 10240 -> 320, up: 320 -> 10240
h      = mean over streams of read * xn              # [tokens, 2560]: the sublayer's input
write  = 2 * sigmoid(inject(xn) / 4)                 # inject: 10240 -> 4, one scalar per stream
x      = x + sublayer(h) * write                     # added to every stream, scaled by its write gate
```

The model starts with 4 identical copies of the token embedding, and ends with one more mixer (`output_hc_*`) that has no write gate. Its `h` goes straight to `lm_head`, with no `output_norm`.

**PLE** (`ple_*`, layer index 1 only), before the layer's attention mixer: `x = x + ple(x, token ids)`.

```
history = previous 2 tokens + these tokens, reset to EOS (248044) after every EOS
for n in 2, 3:                            # bigrams, then trigrams; 8 heads each
  mixed  = t[i]*m0 XOR t[i-1]*m1 (XOR t[i-2]*m2)   # int64 products, wrapping on overflow
  row[h] = (mixed mod head_vocab_sizes[h]) + head_offsets[h]
e      = concat of the 16 rows               # [tokens, 2560]
key    = rms_norm(ple_key(e)) per stream     # 2560 -> 10240
value  = ple_value(e)                        # 2560 -> 2560
gate   = (key · rms_norm(x)) / sqrt(2560) per stream, then signed sqrt: sign(g) * sqrt(|g|)
v      = sigmoid(gate) * value               # [tokens, 4, 2560]
out    = v + silu(conv1d(rms_norm(v)))       # depthwise, kernel 4, dilation 3: 9 inputs of state
```

`layer_multipliers`, `head_offsets` and `head_vocab_sizes` are in the GGUF, so they don't need recomputing.

**QSA indexer** (full-attention layers, `indexer.*`). Keys are pooled over blocks of 4 visible tokens, the mean, then normed and roped at the block's first position. Each block is scored by `sum over the 4 heads of relu(q · k) / sqrt(128)`. Attention keeps the top 512 blocks, plus the tail that doesn't fill a block, and masks the rest.

**Gated DeltaNet**: the output norm is `norm(out) * w * sigmoid(z)` (config `output_gate_type`), where Qwen3.6 uses `silu(z)`.

## References

- transformers `qwen4_exp` at `f210a32`: `modular_qwen4_exp.py` (the readable diff over Qwen3.5), `modeling_qwen4_exp.py` and `configuration_qwen4_exp.py`.
- llama.cpp at `09338b8`: `src/models/qwen4exp.cpp` (graph and hparams), `ggml/src/ggml-quants.c` (Q2_0) and `gguf-py/gguf/quants.py` (numpy dequantizers, for checks).
- The [Unsloth model card](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): architecture summary and sampling settings. Its vLLM link (PR #34330) is an unrelated multimodal change.

## Open questions

- Is IQ → int4 at pack time accurate enough? Q2_0 converts almost exactly, but IQ2_S (layers 8–10) and the 4-bit IQ types gain about 7% relative error (see step 2).
- Does llama.cpp's converter change anything else that needs undoing (V-head order in the GDN layers, `ssm_a` sign, conv shapes), as it does for Qwen3.6? Check against `qwen4exp.cpp`'s tensor shapes.
