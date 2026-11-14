# willow

Run **Qwen3.6-35B-A3B** as a Q4_K_M GGUF on a 16 GB laptop, by keeping the model's experts on the SSD and streaming in only the ones each token needs.

> **Heads up:** this is a vibe-coded proof of concept. It's only been tested on a Mac (M5 MacBook Pro, 16 GB RAM), with one GGUF file. I don't know how well it works on Linux or Windows.

## The model under test

Everything here is built, tuned and tested against one file:

**[`Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive)**

- A fine-tune of [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): a Mixture-of-Experts model with 35B parameters in total and about 3B used per token.
- GGUF architecture `qwen35moe`, made by llama.cpp.
- **Q4_K_M** quantization, which mixes **Q4_K** and **Q6_K** tensors. Those are exactly the two types willow has fast paths for (see [How the GGUF is run](#how-the-gguf-is-run)).
- Text only: the vision encoder and multi-token-prediction head aren't used.

Other GGUFs of the same architecture should work the same way, but this is the only one that has actually been run. See [Other models and formats](#other-models-and-formats).

## Quick start

You need Python 3.9 or newer, and free disk space for the GGUF plus its expert pack (about the same size again, since the experts are most of the model). On a Mac, also install Xcode's Command Line Tools if you haven't (`xcode-select --install`); the GPU path compiles a small helper with `clang++`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the GGUF, for example into `~/Models/`:

```bash
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive \
  Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf --local-dir ~/Models
```

Build the expert pack. You only do this once (and again after upgrading PyTorch). It's written next to the GGUF, in `<name>.expert_pack/`:

```bash
.venv/bin/python main.py pack ~/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
```

Then chat with it:

```bash
.venv/bin/python main.py chat ~/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf --cache-gb 1
```

## Commands

All of these take the `.gguf` file the pack was built from, or a model's name from `models.json` (see below). For brevity, `MODEL` below stands for `~/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`.

**`models.json`** lists models by name, next to `main.py`:

```json
{
  "default": "Qwen3.8-Flash-Next",
  "models": [
    {"name": "Qwen3.8-Flash-Next", "path": "/path/to/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf", "cache_gb": 4},
    {"name": "Qwen3.6-35B-A3B", "path": "/path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf"}
  ]
}
```

Each model can also set `pack_dir`, `cache_gb` and `backbone`; a flag given on the command line wins over them. Names work anywhere a path does (`main.py chat Qwen3.6-35B-A3B`, any case). `main.py web` with no model starts with `default` (or the first model), and the web page can switch between all of them.

**One prompt, one reply:**

```bash
.venv/bin/python main.py run MODEL "Write a short poem about a lazy cat."
.venv/bin/python main.py run MODEL "Write a short poem about a lazy cat." --cache-gb 4   # with a 4 GiB expert cache
```

At the end it prints stats: tokens/s, GiB read per token, disk speed, and where the time went.

**Chat in the terminal:**

```bash
.venv/bin/python main.py chat MODEL --cache-gb 1
```

The model loads once, then you type messages at the `>` prompt. The whole conversation stays in the model's caches, so each reply only has to read your new message, not the conversation again. `/think` turns thinking on or off, `/reset` starts a new conversation, and `/exit` or Ctrl-D quits. Ctrl-C also quits, even mid-reply, since a reply cut off halfway leaves the caches half-updated. After each reply it prints that reply's speed and the conversation's length in tokens.

**Chat in a web page:**

```bash
.venv/bin/python main.py web MODEL --cache-gb 1
```

Once the model has loaded, it opens `http://127.0.0.1:8000`. Use `--port` to pick another port, or `--no-browser` to open the page yourself. Replies stream in as they're written. **Stop** ends a reply after the current token and keeps the conversation going, **New chat** starts over, and the **Think** button toggles thinking. Reloading the page brings back the conversation so far. The model menu at the top switches to another model from `models.json`. That frees the current model first (two don't fit in memory together), so it takes as long as starting willow with that model, and it starts a new chat. If the new model fails to load, the page says why and goes back to the previous one. The page is only reachable from this computer and handles one reply at a time; Ctrl-C in the terminal quits. Nothing extra needs installing: the page is `web/index.html`, served by `web/server.py` with Python's standard library.

Decoding is greedy: it always picks the most likely next token.

### Options for `run`, `chat` and `web`

(`chat` and `web` take no prompt; `--max-new-tokens` applies to each reply.)

| Option | Default | What it does |
|---|---|---|
| `--cache-gb N` | 0 | RAM for caching recently used experts. The cache is memory shared with the GPU, with at least 64 slots (about 120 MiB) even at 0. |
| `--preload` / `--no-preload` | on | Read each layer's most used 15% of experts (from the usage table) into the cache before generating. Needs `--cache-gb`. |
| `--think` | off | Let the model think before answering. This adds a lot of tokens. The thinking is printed dimmed before the answer (in `web`, folded under "Thought"). |
| `--max-new-tokens N` | 100000 | Upper limit on the reply length. Generation stops earlier once the model finishes its answer. |
| `--min-expert-weight W` | 0 (off) | Skip routed experts below routing weight `W`, e.g. `0.02`, keeping each token's top expert and scaling the other weights back up. Faster, but it changes the output and can lower its quality. The stats show how many were skipped. |
| `--gpu-backbone` / `--no-gpu-backbone` | on | With a GPU pack, also run attention, the routers, the shared experts and the output layer on the GPU. `--no-gpu-backbone` keeps them on the CPU. |
| `--pack-dir DIR` | `<name>.expert_pack` | Use a pack stored somewhere else. |
| `--sync-timings` | off | Wait for the GPU at every section boundary, so `decode time per token, by section` shows where GPU time goes. Slower. Without it, GPU work runs in the background and is counted in whichever section next waits for it (mostly `gpu wait`). |

### Options for `pack`

| Option | Default | What it does |
|---|---|---|
| `--out-dir DIR` | `<name>.expert_pack` | Where to write the pack. |
| `--gpu` / `--no-gpu` | on when PyTorch can use MPS | Store the 4-bit experts so they run on the GPU. `--no-gpu` keeps them on the CPU. |

## How it works

The model won't fit in 16 GB of RAM. Most of its size is in the **experts**: 40 layers × 256 experts. For each token, a small router in each layer picks just 8 of its 256 experts, so only a small slice of them is needed at any moment.

1. **Pack:** `main.py pack` copies every expert's weights out of the GGUF into one file, `experts.pack`. Each expert sits in a single 16 KB-aligned block (Apple Silicon's page size), so it can be loaded with one read. `manifest.json` records where each expert is.
2. **Run:** everything except the experts stays in RAM: attention, the routers, the shared experts, the embeddings and the output layer, about 7 GB. At each layer, once the router has picked its 8 experts, all 8 are read from `experts.pack` at once on a pool of threads. Cached experts are computed first while the misses are read, and each miss is computed as soon as its read finishes. While decoding, each layer also runs the next layer's router on its hidden state and starts reading that layer's 4 most likely experts early. macOS's file cache keeps recently read experts in spare RAM too.

Most of the time is still spent reading experts from the SSD.

### How the GGUF is run

The embedding table stays quantized, and rows are dequantized as they're looked up. The other non-expert weights are dequantized when the model loads: to bf16 for Qwen3.6, and to int8 with one scale per row for Qwen3.8 (half the memory, and torch's int8 matrix multiply is faster than bf16 on Apple Silicon). `--backbone int8` or `--backbone bf16` overrides the default.

The experts stay quantized. Turning ggml's quantized blocks back into bf16 with PyTorch takes far longer than reading them from the SSD, so `pack` stores Q4_K_M's two types in layouts that are cheap to run:

- **Q4_K** is stored in PyTorch's own 4-bit format. Each matrix multiply runs straight on the 4-bit weights, so these experts are never dequantized.
- **Q6_K** is requantized to the same 4-bit format when packing. It loses some of Q6_K's precision, but every expert then runs on the 4-bit kernel and is read straight into its cache slot. Packs built earlier stored Q6_K as int8; they still run, but print a note asking you to rebuild.

Each expert's gate and up matrices are also fused into one, so computing an expert takes two matrix multiplies instead of three.

The 4-bit formats come from private PyTorch functions, so a pack records the PyTorch version it was built with. If you upgrade PyTorch, rerun `main.py pack`; an older pack still runs, but prints a note asking you to rebuild it.

### On the GPU

By default (when PyTorch can use Apple's GPU through MPS), `pack` stores the 4-bit experts in the layout of PyTorch's GPU 4-bit kernel, and `run` computes them on the GPU. The expert cache is memory the CPU and GPU share (Apple Silicon has one pool of RAM for both), so each expert is read from the SSD straight into memory the GPU computes from, with no extra copy. Even with `--cache-gb 0`, there are 64 such slots (about 120 MiB) to read into.

The rest of the model runs on the GPU too, so the hidden state stays there. The non-expert weights are moved to GPU memory when the model loads (their CPU copies are freed, so it costs no extra RAM). On the GPU, every small operation costs about the same to issue however little work it does, and each layer has about a hundred of them, so while decoding the small steps are compiled with `torch.compile`, which fuses them into a few GPU kernels. That compile takes about a second on the first generated token, less on later runs (PyTorch caches it on disk). The KV cache grows in blocks of 256 positions, because each new tensor shape costs a fresh GPU setup. Each layer waits for the GPU once, to bring back which experts its tokens picked (and the guesses for the next layer), which the CPU needs to read them.

PyTorch has no Python API for shared GPU memory, so a small Objective-C++ helper (`gguf/mps_shared.mm`) is compiled with `clang++` the first time a GPU pack runs, into `gguf/build/`. A pack's layout decides where its experts run, so to switch, rerun `pack` with `--gpu` or `--no-gpu`.

### Expert cache, usage table and preload

By default no experts are cached in RAM beyond the OS file cache; `--cache-gb N` sets aside N GiB for an expert cache. About three quarters of it holds the **hot set**: the most used experts according to the usage table (below), picked across all layers, so layers whose routing is more concentrated get more. They're only evicted when nothing else can be. The rest of the cache holds everything else that's read (misses and prefetches) and reuses its slots least recently used first. Plain least-recently-used over the whole cache doesn't work when a token reads more experts than the cache holds, which happens with Qwen3.8 (about 500 per token): each token would evict the whole cache before anything is reused. Replaying sampled routing on Qwen3.8's pack, the hit rate went from 0% to 14% at 1 GiB, 23% at 2 GiB and 39% at 4 GiB (somewhat optimistic, since the routing was sampled from the same table).

Each run also keeps a **usage table**: how many tokens each expert in each layer has been routed to, counting both prompt and generated tokens. It's saved to `expert_usage.json` in the pack folder after every run, and each run adds to the counts from earlier ones, so over time it records which experts this model uses most. When the model loads, the hot set is read into the cache before generating, so it starts as cache hits (without slabs, i.e. a pack that isn't int4, it's each layer's most used 15% instead). Experts that have never been used are skipped, and the preload never goes past the cache size. The first run on a pack has no table yet, so it preloads nothing.

## Other models and formats

These load, but haven't been tested the way the Q4_K_M file above has:

- **Other GGUFs** of the `qwen35moe` or `qwen3moe` (the older Qwen3-MoE) architectures, single or split (pass any one of the split files). Supported quant types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q2_0, IQ4_NL, IQ4_XS, IQ3_XXS, IQ2_S. In the experts, Q5_0, IQ4_NL, IQ4_XS, IQ3_XXS, IQ2_S and Q2_0 (Qwen3.8's types) are requantized to the 4-bit format when packing, with a per-group search that keeps the added error low (see ROADMAP.md). Any other type is copied into the pack byte for byte and dequantized as each expert is read, which is slower.

willow only runs GGUF files. The original bf16 safetensors path has been removed; see [ROADMAP.md](ROADMAP.md) for what's coming next.

## Project layout

- `main.py`: the command line.
- `model_list.py`: reads `models.json` and turns a model's name into its path and settings.
- `runner.py`: the terminal side of `run` and `chat`: streaming the reply and printing the stats.
- `gguf/`: everything about the file format, shared by every model: the GGUF file reader (`reader.py`); dequantization (with the IQ grids in `iq_grids.py`), the Q4_K/Q6_K pack formats and the CPU and GPU 4-bit kernels (`quants.py`); the shared CPU/GPU memory helper (`mps_shared.py`, `mps_shared.mm`); the expert-pack compiler (`compiler.py`); the expert store that streams experts from the pack, with its cache, usage table and preload (`store.py`); the tokenizer rebuilt from the GGUF's vocabulary (`tokenizer.py`); and tables read row by row from the GGUF instead of loaded (`streamed.py`).
- `engines/`: one package per model family, picked by the GGUF's `general.architecture` (`engines/__init__.py`). `engines/base.py` holds what they share: the expert streaming, attention and KV cache, the Gated DeltaNet layers, Qwen's chat format and the generate loop.
  - `engines/qwen36/`: Qwen3.6 (and Qwen3.5 and Qwen3-MoE). How their layers are wired (`model.py`) and its GGUF loader (`loader.py`), which renames the GGUF tensors to their Hugging Face names and undoes llama.cpp's conversion changes.
- `web/`: the chat page for `main.py web`: the page (`index.html`) and its server (`server.py`). Replies are rendered as Markdown with [marked](https://github.com/markedjs/marked) 16.4.2, kept in `web/vendor/` (MIT license, `marked-LICENSE.md`) so the page works offline.

## Checking correctness

The reference checks compared the engine against the original safetensors model and Hugging Face transformers, so they were removed along with the safetensors path. A GGUF-only check is on the [roadmap](ROADMAP.md).
