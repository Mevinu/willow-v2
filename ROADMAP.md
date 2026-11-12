# Roadmap: Qwen3.8-Flash-Next

Goal: run **Qwen3.8-Flash-Next** (`Qwen3.8-Flash-Next-UD-IQ4_XS`, Unsloth's GGUF, 4 split files, about 87 GiB) on the same 16 GB Mac, streaming its experts from the SSD the way willow runs Qwen3.6. Qwen3.6 must keep working the whole way through.

## What the new model is

The GGUF's architecture is `qwen4exp` ("A Preview of the Qwen4 Architecture"). Everything below comes from its header; nothing has been run yet.

| | Qwen3.6-35B-A3B (today) | Qwen3.8-Flash-Next |
|---|---|---|
| Architecture | `qwen35moe` | `qwen4exp` |
| Layers | 40 | 48 |
| Hidden size | 2048 | 2560 |
| Experts | 256, 8 per token | 512, **10** per token, expert FFN 640 |
| Shared expert | yes, sigmoid-gated | yes, sigmoid-gated (FFN 640) |
| Token mixers | Gated DeltaNet ×3, then full attention | the same pattern: 36 linear layers, 12 full (every 4th) |
| Full attention | gated, GQA | gated, 24 heads / 2 KV heads, head dim 256, **KV compression ratio 4**, **sparse indexer** (4 heads, top-k 2048) |
| Residual stream | one 2048-wide stream | **hyper-connections**: 4 streams (10240 wide), low-rank 320 mixing per sublayer |
| Norms | `attn_norm`, `post_attention_norm`, `output_norm` | none of these: `hc_attn_norm`, `hc_ffn_norm`, `output_hc_norm` instead |
| Extra embeddings | none | **per-layer n-gram embeddings (PLE)** on layer 1: a hashed 3-gram table with 320M rows × 160 dims (16 heads, prime-sized buckets), with a conv, key/value and norms |
| Quant types | Q4_K, Q6_K | IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_S, Q5_0, Q6_K, Q8_0, BF16, F32 and **ggml type 42** (unknown; 2.25 bits/weight) |
| Tokenizer and chat format | `gpt2` BPE, `qwen35` pre-tokenizer, `<think>` | the same |
| Sampling defaults | (greedy in willow) | temp 1.0, top-k 20, top-p 0.95 |

Sizes on disk: experts 56.2 GiB, the PLE table 26.8 GiB, everything else 4.2 GiB.

## What this means for willow

- **The GGUF reader rejects the file today.** Type 42 is unknown to it, and nothing downstream handles IQ types.
- **Per-token I/O roughly doubles.** That's 480 experts per token (48 × 10), against 320 for Qwen3.6. An expert is about 2.3 MiB as stored, against 1.9 MiB, so a token that misses every cache reads about 1.1 GiB (Qwen3.6: about 0.6 GiB). The usage table, preload and next-layer prefetch matter even more here.
- **The PLE table can't live in RAM** (26.8 GiB). It has to be streamed too: each token looks up 16 rows of 90 bytes, so that's random 16 KB page reads, small next to the experts.
- **RAM is tight.** Dequantized to bf16 like Qwen3.6's, the non-expert weights take about 8 GiB (plus 1.2 GiB for the embedding table, if it can't stay quantized). Some weights may need to stay 8-bit.
- **The forward pass is new.** Hyper-connections change how every layer reads and writes the residual stream, so `MoEEngine.forward` can't be reused as is. The Gated DeltaNet layers, the expert streaming (`moe`), the KV cache, the GPU paths and the generate loop are all worth sharing.
- **There is no local reference implementation.** Every formula for hyper-connections, PLE, KV compression and the indexer has to come from source code, not guesses. This is the biggest risk.

## Plan

### 0. Isolate Qwen3.6, remove safetensors ✅

- [x] Remove the safetensors path: its loader, its pack compiler and its reference checks, plus `requirements-ref.txt` and the `safetensors`/`transformers` dependencies.
- [x] Split the code into `gguf/` (format: reader, quants, pack, expert store, tokenizer), `engines/qwen36/` (Qwen3.6 model and loader) and `runner.py` (CLI loops). `engines/__init__.py` picks an engine from `general.architecture`.
- [ ] Confirm Qwen3.6 gives the same output as before the split (same greedy reply, same speed).

### 1. Find the reference

- [ ] Find the llama.cpp source (fork, PR or branch) that wrote this GGUF: the `qwen4exp` graph, and ggml type 42 (its name, block layout and dequantization).
- [ ] If they exist, find the Hugging Face modeling code / config for Qwen3.8-Flash-Next.
- [ ] Write down the exact math for: hyper-connections (the norm, down/up and inject, and how the 4 streams are mixed and combined at the output), PLE (the hash with `layer_multipliers`, `head_offsets` and `head_vocab_sizes`, then conv, key/value and how the result enters layer 1), KV compression (ratio 4) and the sparse indexer.
- [ ] Set up a way to get reference logits for a short prompt (e.g. llama.cpp's own build of this model), so every later step can be checked against it.

### 2. Format support (`gguf/`)

- [ ] Reader: add type 42 and make unknown types fail with a clear message.
- [ ] Dequantizers for IQ4_NL, IQ4_XS, IQ3_XXS, IQ2_S and type 42 in `quants.py` (IQ3_XXS and IQ2_S need ggml's grid tables), checked against llama.cpp on real tensors.
- [ ] `QuantizedEmbedding` for IQ4_XS, so `token_embd` stays quantized.
- [ ] A pack layout for IQ experts. Choose between:
  - (a) requantizing to INT4_G32 at pack time, as Q6_K is today. It reuses the int4 GPU kernel, but the pack grows to about 5 bits/weight (about 1.4 GiB/token cold) and the 2–3-bit layers lose a little.
  - (b) keeping the IQ bytes and dequantizing per read, which is simple but slow.
  - (c) a Metal kernel that runs IQ blocks directly. Fastest and smallest, but the most work.
  Start with (a) and measure.
- [ ] A GGUF-only regression check that replaces the deleted reference checks: a short prompt's logits and greedy reply saved as a golden file per model.

### 3. The `qwen4exp` engine (`engines/qwen38/`)

- [ ] Pull out the parts that don't depend on the model and both engines need: `moe` streaming and routing, Gated DeltaNet, the attention core and KV cache, `Ops`, and the generate/stats loop, into a shared base. Qwen3.6's output must not change.
- [ ] Loader: config from `qwen4exp.*`, tensor names, and any changes llama.cpp's converter made that need undoing (check whether the `-1` norm and V-head reorder fixes from Qwen3.6 apply).
- [ ] Hyper-connection residual stream.
- [ ] Gated full attention with KV compression. Skip the sparse indexer at first: with top-k 2048 it selects everything until the context passes 2048 (compressed) positions, so dense attention gives the same result for short contexts.
- [ ] PLE: hashed n-gram lookups streamed from the GGUF (pread or mmap, never resident), plus the PLE block on layer 1.
- [ ] Routing for 512 experts / top-10 (check for normalization and any bias or scaling in the router).
- [ ] Chat: the same format as Qwen3.6, so `chat` and `web` should work unchanged.
- [ ] Match the reference logits layer by layer, then end to end.

### 4. Make it fast on 16 GB

- [ ] A memory budget: decide which non-expert weights stay 8-bit, and how big the expert cache can be.
- [ ] GPU paths: `torch.compile` for the hyper-connection and PLE steps, and a shared-memory slab for the 512-expert layers.
- [ ] Tune prefetch and preload for top-10 of 512 (the usage table probably matters more with this many experts).
- [ ] The sparse indexer, for contexts past its top-k.
- [ ] Optional: sampling (temp, top-k, top-p) using the GGUF's defaults, alongside greedy.

## Open questions

- What is ggml type 42, and which llama.cpp build reads it?
- How exactly do hyper-connections mix the 4 streams (static and dynamic parts, and how `inject` is used)?
- Is PLE added to the hidden state, to one stream, or gated in some other way? Why does it have its own EOS token (`ple.eos_token_id` = 248044)?
- What does a compression ratio of 4 on full-attention layers do to the KV cache (pooling? a learned compressor?), and does it change RoPE positions?
- Is IQ → int4 at pack time accurate enough for the IQ2_S layers (8–10)?
