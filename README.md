# willow

Run a large Mixture-of-Experts (MoE) language model on a laptop that doesn't have enough RAM to hold it, by streaming the model's experts from the SSD.

> **Heads up:** this is a vibe-coded proof of concept. I've only tested it on a Mac (M5 MacBook Pro, 16 GB RAM), so I don't know how well it works on Linux or Windows.

## Target model

It's built and tuned for **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)**: 35B parameters in total, about 3B used per token. It runs the original bf16 weights, so nothing is quantized and you get the model's full quality. The vision encoder and multi-token-prediction head aren't used; this is text only.

The engine also supports the older Qwen3-MoE models (`model_type: qwen3_moe`).

On the 16 GB M5 MacBook Pro, it generates about **3 tokens/s** and processes prompts at about **6 tokens/s**.

## How it works

The model's weights are about 67 GB, which won't fit in 16 GB of RAM. Most of that size is in the **experts**: 40 layers × 256 experts, about 60 GB in total. For each token, a small router in each layer picks just 8 of its 256 experts, so only a small slice of them is needed at any moment.

1. **Pack:** `main.py pack` copies every expert's weights, byte for byte, out of the safetensors files into one file, `experts.pack`. In it, each expert sits in a single 4 KB-aligned block, so it can be loaded with one read. `manifest.json` records where each expert is.
2. **Run:** `main.py run` keeps everything except the experts in RAM: attention, the router, the shared expert, the embeddings and the output layer. That's about 7 GB. At each layer, once the router has picked its 8 experts, all 8 are read from `experts.pack` at the same time on a pool of threads. Each one is computed as soon as its read finishes. macOS's file cache keeps recently read experts in spare RAM automatically.

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

Options for `run`:

| Option | Default | What it does |
|---|---|---|
| `--max-new-tokens N` | 2048 | Upper limit on the reply length. Generation stops earlier once the model finishes its answer. |
| `--think` | off | Lets the model think before answering. This adds a lot of tokens. |
| `--pack-dir DIR` | `<model_dir>/expert_pack` | Use a pack stored somewhere else. |
| `--cache-gb N` | 0 | RAM for an extra cache of recently used experts. Leave it at 0 on a 16 GB machine: the OS file cache already does this, and an extra cache causes memory pressure that slows everything down. |

Decoding is greedy: it always picks the most likely next token.

## Checking correctness (optional)

- `check_engine.py` compares the engine's output with Hugging Face transformers on a small Qwen3-MoE model. It needs the `transformers<5` package from `requirements.txt`.
- `check_qwen35_layers.py` compares individual Qwen3.6 layers against transformers' own implementation, using the real weights. It needs transformers 5, which needs Python 3.10+. See `requirements-ref.txt`:

  ```bash
  uv venv .venv-ref --python 3.12 && VIRTUAL_ENV=.venv-ref uv pip install -r requirements-ref.txt
  .venv-ref/bin/python check_qwen35_layers.py ~/Models/Qwen3.6-35B-A3B 0 3 39
  ```
