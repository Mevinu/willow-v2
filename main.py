import argparse
import os
import sys

from model_list import read_models, resolve

MODEL_HELP = "A .gguf file (for a split GGUF, any one of its files), or a model's name in models.json"

def check_gguf(model: str):
    if not model.endswith(".gguf"):
        sys.exit(f"{model} is not a .gguf file (safetensors folders are no longer supported)")
    if not os.path.exists(model):
        sys.exit(f"{model} doesn't exist")

def add_run_options(command):
    """The options `run` and `chat` share."""
    command.add_argument("--max-new-tokens", type=int, default=100000,
                         help="upper limit (per reply, for chat); generation stops earlier when the model emits an "
                              "end token")
    command.add_argument("--pack-dir", default=None,
                         help="Default: the model's pack_dir in models.json, else where `main.py pack` writes it")
    command.add_argument("--think", action="store_true", help="Let chat models think before answering")
    command.add_argument("--cache-gb", type=float, default=None,
                         help="GiB of RAM for caching recently used experts (default: the model's cache_gb in "
                              "models.json, else 0, off; the OS file cache still keeps recently read experts in "
                              "spare RAM). Needed for --preload")
    command.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                         help="Read each layer's most used 15%% of experts (from earlier runs) into the "
                              "cache before generating (needs --cache-gb)")
    command.add_argument("--gpu-backbone", action=argparse.BooleanOptionalAction, default=True,
                         help="With a GPU pack: also run attention, the router, the shared expert and the "
                              "output layer on the GPU (default: on; --no-gpu-backbone keeps them on the CPU)")
    command.add_argument("--sync-timings", action="store_true",
                         help="Wait for the GPU at each section boundary, so 'decode time per token, by section' "
                              "shows where GPU time goes (slower; without it, GPU work is counted wherever the CPU "
                              "next waits)")
    command.add_argument("--min-expert-weight", type=float, default=0.0,
                         help="Skip routed experts below this routing weight, e.g. 0.02 (default: 0, "
                              "off). Faster, but it changes the output and can lower its quality")
    command.add_argument("--backbone", choices=("int8", "bf16"), default=None,
                         help="How the non-expert weights are held in memory: int8 (half the memory, slightly less "
                              "precise) or bf16 (default: the model's backbone in models.json, else int8 for "
                              "Qwen3.8 and bf16 for Qwen3.6)")

def main():
    parser = argparse.ArgumentParser(description="Run large MoE models by streaming their experts from SSD")
    commands = parser.add_subparsers(dest="command", required=True)

    pack = commands.add_parser("pack", help="Compile a model's MoE expert weights into a 16KB-aligned pack file")
    pack.add_argument("model", help=MODEL_HELP)
    pack.add_argument("--out-dir", default=None,
                      help="Where to write the pack (default: <model>.expert_pack next to the .gguf file)")
    pack.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                      help="Store the int4 experts in the GPU (MPS) kernel's format so they run on the "
                           "GPU (default: on when torch can use MPS; --no-gpu keeps them on the CPU)")

    run = commands.add_parser("run", help="Generate text from a prompt, streaming experts from the pack")
    run.add_argument("model", help=MODEL_HELP)
    run.add_argument("prompt")
    add_run_options(run)

    chat = commands.add_parser("chat", help="Chat with the model; it remembers the conversation")
    chat.add_argument("model", help=MODEL_HELP)
    add_run_options(chat)

    web = commands.add_parser("web", help="Chat in a web page on this computer; it remembers the "
                                          "conversation")
    web.add_argument("model", nargs="?", default=None,
                     help=MODEL_HELP + " (default: models.json's default). The page can switch to any model in "
                          "models.json")
    add_run_options(web)
    web.add_argument("--port", type=int, default=8000, help="Port for the page on 127.0.0.1 (default: 8000)")
    web.add_argument("--no-browser", action="store_true", help="Don't open the page in a browser")
    args = parser.parse_args()

    models, default = read_models()
    entry = resolve(args.model, models, default)
    check_gguf(entry["path"])
    if args.command == "pack":
        from gguf.compiler import compile_pack
        compile_pack(entry["path"], args.out_dir or entry.get("pack_dir"), args.gpu)
        return

    from engines import load_engine

    def load_model(entry: dict):
        """Loads a models.json entry. Command-line flags win over the entry's settings."""
        check_gguf(entry["path"])
        backbone = args.backbone or entry.get("backbone")
        options = {"backbone": backbone} if backbone else {}
        cache_gb = args.cache_gb if args.cache_gb is not None else entry.get("cache_gb", 0)
        return load_engine(entry["path"], args.pack_dir or entry.get("pack_dir"), cache_gb, preload=args.preload,
                           gpu_backbone=args.gpu_backbone, sync_timings=args.sync_timings,
                           min_expert_weight=args.min_expert_weight, **options)

    if args.command == "web":
        from web.server import run_web
        if entry not in models:  # a .gguf file models.json doesn't list: switchable to as well
            models = [entry] + models
        run_web(models, entry["name"], args.max_new_tokens, args.think, load_model, args.port, not args.no_browser)
        return
    from runner import run_chat, run_prompt
    if args.command == "chat":
        run_chat(lambda: load_model(entry), args.max_new_tokens, args.think)
    else:
        run_prompt(lambda: load_model(entry), args.prompt, args.max_new_tokens, args.think)

if __name__ == "__main__":
    main()
