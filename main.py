import argparse

def is_gguf(model: str) -> bool:
    return model.endswith(".gguf")

def add_run_options(command):
    """The options `run` and `chat` share."""
    command.add_argument("--max-new-tokens", type=int, default=100000,
                         help="upper limit (per reply, for chat); generation stops earlier when the model emits an "
                              "end token")
    command.add_argument("--pack-dir", default=None, help="Default: where `main.py pack` writes it")
    command.add_argument("--think", action="store_true", help="Let chat models think before answering")
    command.add_argument("--cache-gb", type=float, default=0,
                         help="GiB of RAM for caching recently used experts (default: 0, off; the OS file cache "
                              "still keeps recently read experts in spare RAM). Needed for --preload")
    command.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                         help="For .gguf: read each layer's most used 15%% of experts (from earlier runs) into the "
                              "cache before generating (needs --cache-gb)")
    command.add_argument("--gpu-backbone", action=argparse.BooleanOptionalAction, default=True,
                         help="For .gguf with a GPU pack: also run attention, the router, the shared expert and the "
                              "output layer on the GPU (default: on; --no-gpu-backbone keeps them on the CPU)")
    command.add_argument("--sync-timings", action="store_true",
                         help="Wait for the GPU at each section boundary, so 'decode time per token, by section' "
                              "shows where GPU time goes (slower; without it, GPU work is counted wherever the CPU "
                              "next waits)")
    command.add_argument("--min-expert-weight", type=float, default=0.0,
                         help="For .gguf: skip routed experts below this routing weight, e.g. 0.02 (default: 0, "
                              "off). Faster, but it changes the output and can lower its quality")

def main():
    parser = argparse.ArgumentParser(description="Run large MoE models by streaming their experts from SSD")
    commands = parser.add_subparsers(dest="command", required=True)

    pack = commands.add_parser("pack", help="Compile a model's MoE expert weights into a 16KB-aligned pack file")
    pack.add_argument("model", help="Hugging Face model folder containing .safetensors files, or a .gguf file "
                                    "(for a split GGUF, any one of its files)")
    pack.add_argument("--out-dir", default=None,
                      help="Where to write the pack (default: <model_dir>/expert_pack, or <model>.expert_pack "
                           "next to a .gguf file)")
    pack.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                      help="For .gguf: store the int4 experts in the GPU (MPS) kernel's format so they run on the "
                           "GPU (default: on when torch can use MPS; --no-gpu keeps them on the CPU)")

    run = commands.add_parser("run", help="Generate text from a prompt, streaming experts from the pack")
    run.add_argument("model", help="Model folder or .gguf file (the one `main.py pack` was run on)")
    run.add_argument("prompt")
    add_run_options(run)

    chat = commands.add_parser("chat", help="Chat with the model; it remembers the conversation (Qwen3.5/3.6)")
    chat.add_argument("model", help="Model folder or .gguf file (the one `main.py pack` was run on)")
    add_run_options(chat)

    web = commands.add_parser("web", help="Chat in a web page on this computer; it remembers the conversation "
                                          "(Qwen3.5/3.6)")
    web.add_argument("model", help="Model folder or .gguf file (the one `main.py pack` was run on)")
    add_run_options(web)
    web.add_argument("--port", type=int, default=8000, help="Port for the page on 127.0.0.1 (default: 8000)")
    web.add_argument("--no-browser", action="store_true", help="Don't open the page in a browser")
    args = parser.parse_args()

    if args.command == "pack":
        if is_gguf(args.model):
            from gguf.compiler import compile_pack
            compile_pack(args.model, args.out_dir, args.gpu)
        else:
            from safetensor.compiler import compile_pack
            compile_pack(args.model, args.out_dir)
        return

    # Imported here so `pack` works without torch installed
    from safetensor.engine import run_chat, run_prompt
    if is_gguf(args.model):
        from functools import partial
        from gguf.engine import load_engine
        load_engine = partial(load_engine, preload=args.preload, gpu_backbone=args.gpu_backbone,
                              sync_timings=args.sync_timings, min_expert_weight=args.min_expert_weight)
    else:
        from safetensor.engine import load_engine
    if args.command == "web":
        from web.server import run_web
        run_web(args.model, args.max_new_tokens, args.pack_dir, args.think, args.cache_gb, load_engine,
                args.port, not args.no_browser)
    elif args.command == "chat":
        run_chat(args.model, args.max_new_tokens, args.pack_dir, args.think, args.cache_gb, load_engine)
    else:
        run_prompt(args.model, args.prompt, args.max_new_tokens, args.pack_dir, args.think, args.cache_gb,
                   load_engine)

if __name__ == "__main__":
    main()
