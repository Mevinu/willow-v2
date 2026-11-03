import argparse

from expert_pack_compiler import compile_pack

def main():
    parser = argparse.ArgumentParser(description="Run large MoE models by streaming their experts from SSD")
    commands = parser.add_subparsers(dest="command", required=True)

    pack = commands.add_parser("pack", help="Compile a model's MoE expert weights into a 4KB-aligned pack file")
    pack.add_argument("model_dir", help="Hugging Face model folder containing .safetensors files")
    pack.add_argument("--out-dir", default=None, help="Where to write the pack (default: <model_dir>/expert_pack)")

    run = commands.add_parser("run", help="Generate text from a prompt, streaming experts from the pack")
    run.add_argument("model_dir", help="Model folder (the one `main.py pack` was run on)")
    run.add_argument("prompt")
    run.add_argument("--max-new-tokens", type=int, default=2048,
                     help="upper limit; generation stops earlier when the model emits an end token")
    run.add_argument("--pack-dir", default=None, help="Default: <model_dir>/expert_pack")
    run.add_argument("--think", action="store_true", help="Let chat models think before answering")
    run.add_argument("--cache-gb", type=float, default=0.0,
                     help="RAM for caching recently used experts (default off: on a 16 GB Mac the OS file cache "
                          "does better, and an explicit cache causes memory pressure)")
    args = parser.parse_args()

    if args.command == "pack":
        compile_pack(args.model_dir, args.out_dir)
    else:
        # Imported here so `pack` works without torch installed
        from engine import run_prompt
        run_prompt(args.model_dir, args.prompt, args.max_new_tokens, args.pack_dir, args.think, args.cache_gb)

if __name__ == "__main__":
    main()
