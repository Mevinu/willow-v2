# Contributing to willow

Thanks for looking at willow. It's a small, experimental project, so small focused contributions are the most useful.

## Good first contributions

- **Test on other hardware.** Run the quick start on Linux, Windows, or a different Mac, and open an issue with what worked and what didn't. The end of each `run` prints tokens/s and GiB read per token, which is useful to include.
- **Try another GGUF.** `README.md` lists the architectures and quant types that load. Reports from models other than the Q4_K_M file are very welcome.
- **Docs fixes.** If a command in the README doesn't work as written, a fix to the README is a complete PR on its own.
- **Items from [ROADMAP.md](ROADMAP.md).** Check the roadmap before starting anything big, so we don't duplicate work.

## Setting up

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The README's Quick start covers downloading the model and building an expert pack.

## Before opening a PR

- Keep each PR to one change, and say in the description what you tested and on which machine.
- If you change how a command behaves, update the matching section of `README.md`.
- Don't commit model files, expert packs, or a `models.json` with personal paths (these are already in `.gitignore`).

## Reporting bugs

Please include the command you ran, the full error output, your OS and Python/PyTorch versions, and the model file name.
