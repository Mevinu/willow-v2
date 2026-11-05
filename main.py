import argparse

def is_gguf(model: str) -> bool:
    return model.endswith(".gguf")

def main():
    parser = argparse.ArgumentParser(description="Run large MoE models by streaming their experts from SSD")
    commands = parser.add_subparsers(dest="command", required=True)

    pack = commands.add_parser("pack", help="Compile a model's MoE expert weights into a 16KB-aligned pack file")
    pack.add_argument("model", help="Hugging Face model folder containing .safetensors files, or a .gguf file "
                                    "(for a split GGUF, any one of its files)")
    pack.add_argument("--out-dir", default=None,
                      help="Where to write the pack (default: <model_dir>/expert_pack, or <model>.expert_pack "
                           "next to a .gguf file)")

    run = commands.add_parser("run", help="Generate text from a prompt, streaming experts from the pack")
    run.add_argument("model", help="Model folder or .gguf file (the one `main.py pack` was run on)")
    run.add_argument("prompt")
    run.add_argument("--max-new-tokens", type=int, default=2048,
                     help="upper limit; generation stops earlier when the model emits an end token")
    run.add_argument("--pack-dir", default=None, help="Default: where `main.py pack` writes it")
    run.add_argument("--think", action="store_true", help="Let chat models think before answering")
    run.add_argument("--cache-gb", type=float, default=0,
                     help="GiB of RAM for caching recently used experts (default: 0, off; the OS file cache "
                          "still keeps recently read experts in spare RAM). Needed for --preload")
    run.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                     help="For .gguf: read each layer's most used 15%% of experts (from earlier runs) into the "
                          "cache before generating (needs --cache-gb)")
    pack.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                      help="For .gguf: store the int4 experts in the GPU (MPS) kernel's format so they run on the "
                           "GPU (default: on when torch can use MPS; --no-gpu keeps them on the CPU)")
    args = parser.parse_args()

    if args.command == "pack":
        if is_gguf(args.model):
            from gguf.compiler import compile_pack
            compile_pack(args.model, args.out_dir, args.gpu)
        else:
            from safetensor.compiler import compile_pack
            compile_pack(args.model, args.out_dir)
    else:
        # Imported here so `pack` works without torch installed
        from safetensor.engine import run_prompt
        if is_gguf(args.model):
            from functools import partial
            from gguf.engine import load_engine
            load_engine = partial(load_engine, preload=args.preload)
        else:
            from safetensor.engine import load_engine
        run_prompt(args.model, args.prompt, args.max_new_tokens, args.pack_dir, args.think, args.cache_gb, load_engine)

if __name__ == "__main__":
    main()
