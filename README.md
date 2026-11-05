# willow

Run a large Mixture-of-Experts (MoE) language model on a laptop that doesn't have enough RAM to hold it, by streaming the model's experts from the SSD.

> **Heads up:** this is a vibe-coded proof of concept. I've only tested it on a Mac (M5 MacBook Pro, 16 GB RAM), so I don't know how well it works on Linux or Windows.

## Target model

It's built and tuned for **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)**: 35B parameters in total, about 3B used per token. It runs either the original bf16 weights (full quality) or a quantized GGUF file (smaller, so less to read per token). The vision encoder and multi-token-prediction head aren't used; this is text only.

The engine also supports the older Qwen3-MoE models (`model_type: qwen3_moe`).

It runs models in two formats:

- **safetensors**: the Hugging Face model folder, as downloaded.
- **GGUF**: a single `.gguf` file (or a split one) made by llama.cpp, for the `qwen35moe` and `qwen3moe` architectures. The experts stay quantized on disk, so each token reads far less than with bf16. Supported quant types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (IQ quants aren't supported yet).

> **GGUF testing so far:** I've only tested GGUF with one **Q4_K_M** file of Qwen3.6-35B-A3B. Q4_K_M mixes Q4_K and Q6_K, the two types with a fast path (see below). Other quant types should load, but they haven't been tried and go through a slower dequantization path.

On the 16 GB M5 MacBook Pro, with the bf16 safetensors model, it generates about **3 tokens/s** and processes prompts at about **6 tokens/s**.

## How it works

The model's weights are about 67 GB, which won't fit in 16 GB of RAM. Most of that size is in the **experts**: 40 layers × 256 experts, about 60 GB in total. For each token, a small router in each layer picks just 8 of its 256 experts, so only a small slice of them is needed at any moment.

1. **Pack:** `main.py pack` copies every expert's weights out of the safetensors (or GGUF) files into one file, `experts.pack`. In it, each expert sits in a single 16 KB-aligned block (Apple Silicon's page size), so it can be loaded with one read. `manifest.json` records where each expert is.
2. **Run:** `main.py run` keeps everything except the experts in RAM: attention, the router, the shared expert, the embeddings and the output layer. That's about 7 GB. At each layer, once the router has picked its 8 experts, all 8 are read from `experts.pack` at the same time on a pool of threads. Cached experts are computed first, while the misses are read, and each miss is computed as soon as its read finishes. While decoding, each layer also runs the next layer's router on its hidden state and starts reading that layer's 4 most likely experts early. macOS's file cache keeps recently read experts in spare RAM automatically.

For a GGUF model, the non-expert weights are dequantized to bf16 when the model loads, so they still take about 7 GB of RAM (on Apple Silicon, bf16 matrix multiplies are faster for them than PyTorch's 4-bit kernel). The experts stay quantized. By default no experts are cached in RAM beyond the OS file cache; `--cache-gb N` sets aside N GiB for a cache of recently used ones. Turning ggml's quantized blocks back into bf16 with PyTorch takes far longer than reading them from the SSD, so `pack` stores the two most common types in layouts that are cheap to run:

- **Q4_K** is stored in PyTorch's own 4-bit format. Each matrix multiply runs straight on the 4-bit weights, so these experts are never dequantized.
- **Q6_K** is requantized to the same 4-bit format when packing. It loses some of Q6_K's precision, but every expert then runs on the 4-bit kernel and is read straight into its cache slot. Packs built earlier store Q6_K as int8; they still run, but print a note asking you to rebuild.

Each expert's gate and up matrices are also fused into one, so computing an expert takes two matrix multiplies instead of three.
- Other types are copied byte for byte and dequantized as each expert is read, which is slower.

**On the GPU:** by default (when PyTorch can use Apple's GPU through MPS), `pack` stores these 4-bit experts in the layout of PyTorch's GPU 4-bit kernel instead, and `run` computes the experts on the GPU. The expert cache is then memory the CPU and GPU share (Apple Silicon has one pool of RAM for both), so each expert is read from the SSD straight into memory the GPU computes from, with no extra copy. Even with `--cache-gb 0`, there are 64 such slots (about 85 MB) to read into. Each layer's expert outputs are added up on the GPU and copied back once per layer. Everything else (attention, the router, the shared expert) still runs on the CPU. PyTorch has no Python API for shared GPU memory, so a small Objective-C++ helper (`gguf/mps_shared.mm`) is compiled with `clang++` the first time a GPU pack runs, into `gguf/build/`. This needs Xcode or its Command Line Tools (`xcode-select --install`). A pack's layout decides where its experts run, so to switch, rerun `pack` with `--gpu` or `--no-gpu`.

The 4-bit formats come from private PyTorch functions, so a GGUF pack records the PyTorch version it was built with. If you upgrade PyTorch, rerun `main.py pack`. A GGUF pack built before this change still runs, but prints a note asking you to rebuild it.

A GGUF run also keeps a **usage table**: how many tokens each expert in each layer has been routed to, counting both prompt and generated tokens. It's saved to `expert_usage.json` in the pack folder after every run, and each run adds to the counts from the ones before, so over time it records which experts the model uses most. When the model loads, each layer's most used 15% of experts are read into the RAM cache before generating, so they start as cache hits. Experts that have never been used are skipped, and the preload never goes past the cache size. The first run on a pack has no table yet, so it preloads nothing. Preloading needs the cache, so it only happens with `--cache-gb` set; turn it off with `--no-preload`.

At the end of each run, it prints stats: tokens/s, GiB read per token, disk speed, and where the time went. Right now, most of the time is spent reading experts, about 1.9 GiB per token.

## Setup

You need Python 3.9 or newer, and about 130 GB of free disk space: 67 GB for the model and 60 GB for the expert pack.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the model:

```bash
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download Qwen/Qwen3.6-35B-A3B --local-dir ~/Models/Qwen3.6-35B-A3B
```

## Usage

Build the expert pack. You only need to do this once per model. It writes `<model_dir>/expert_pack/`:

```bash
.venv/bin/python main.py pack ~/Models/Qwen3.6-35B-A3B
```

Generate text:

```bash
.venv/bin/python main.py run ~/Models/Qwen3.6-35B-A3B "Write a short poem about a lazy cat."
```

For a GGUF model, pass the `.gguf` file instead of the folder (for a split model, any one of its files). The pack goes next to it, in `<name>.expert_pack/`:

```bash
.venv/bin/python main.py pack ~/Models/Qwen3.6-35B-A3B-Q4_K_M.gguf
.venv/bin/python main.py run ~/Models/Qwen3.6-35B-A3B-Q4_K_M.gguf "Write a short poem about a lazy cat."
```

Options for `pack`:

| Option | Default | What it does |
|---|---|---|
| `--out-dir DIR` | `<model_dir>/expert_pack` or `<name>.expert_pack` | Where to write the pack. |
| `--gpu` / `--no-gpu` | on when PyTorch can use MPS | GGUF only: store the 4-bit experts so they run on the GPU. `--no-gpu` keeps them on the CPU. |

Options for `run`:

| Option | Default | What it does |
|---|---|---|
| `--max-new-tokens N` | 2048 | Upper limit on the reply length. Generation stops earlier once the model finishes its answer. |
| `--think` | off | Lets the model think before answering. This adds a lot of tokens. |
| `--pack-dir DIR` | `<model_dir>/expert_pack` or `<name>.expert_pack` | Use a pack stored somewhere else. |
| `--cache-gb N` | 0 | RAM for caching recently used experts. For safetensors, leave it at 0 on a 16 GB machine: the OS file cache already does this, and an extra cache causes memory pressure that slows everything down. For a GPU pack, the cache is memory shared with the GPU, with at least 64 slots (about 85 MB) even at 0. |
| `--preload` / `--no-preload` | on | GGUF only: read each layer's most used 15% of experts (from the usage table) into the cache before generating. Needs `--cache-gb`. |

Decoding is greedy: it always picks the most likely next token.

## Project layout

- `main.py`: the command line. It picks the GGUF code for a `.gguf` file and the safetensors code for a folder.
- `safetensor/`: the model code (`engine.py`), the safetensors expert-pack compiler (`compiler.py`) and its checks.
- `gguf/`: the GGUF file reader (`reader.py`), dequantization and the Q4_K/Q6_K pack formats and the CPU and GPU 4-bit kernels (`quants.py`), the shared CPU/GPU memory helper (`mps_shared.py`, `mps_shared.mm`), the GGUF expert-pack compiler (`compiler.py`) and the GGUF loader (`engine.py`). The loader renames the GGUF tensors to their Hugging Face names and undoes llama.cpp's conversion changes, then runs them on the model code in `safetensor/engine.py`.

## Checking correctness (optional)

- `gguf/check_engine.py` runs the same model from safetensors and from a GGUF, one after the other, and compares their prompt logits and greedy replies:

  ```bash
  .venv/bin/python -m gguf.check_engine ~/Models/Qwen3.6-35B-A3B ~/Models/Qwen3.6-35B-A3B-Q8_0.gguf
  ```

- `safetensor/check_engine.py` compares the engine's output with Hugging Face transformers on a small Qwen3-MoE model. It needs the `transformers<5` package from `requirements.txt`.
- `safetensor/check_qwen35_layers.py` compares individual Qwen3.6 layers against transformers' own implementation, using the real weights. It needs transformers 5, which needs Python 3.10+. See `requirements-ref.txt`:

  ```bash
  uv venv .venv-ref --python 3.12 && VIRTUAL_ENV=.venv-ref uv pip install -r requirements-ref.txt
  .venv-ref/bin/python -m safetensor.check_qwen35_layers ~/Models/Qwen3.6-35B-A3B 0 3 39
  ```
