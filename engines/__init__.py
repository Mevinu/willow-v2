"""The model architectures willow runs, picked by the GGUF's general.architecture. Each lives in its own package
here, with a loader.load_engine(gguf_path, metadata, tensors, pack_dir, cache_gb, **options)."""
import importlib

from gguf.reader import read_gguf

ARCHITECTURES = {
    "qwen3moe": "engines.qwen36",   # Qwen3-MoE (shares Qwen3.6's code: it's the base engine there)
    "qwen35moe": "engines.qwen36",  # Qwen3.5 / Qwen3.6 MoE
    "qwen4exp": "engines.qwen38",   # Qwen3.8-Flash-Next
}

def load_engine(gguf_path: str, pack_dir: str = None, cache_gb: float = None, **options):
    """Reads the GGUF's header and hands it to its architecture's loader."""
    metadata, tensors = read_gguf(gguf_path)
    arch = metadata.get("general.architecture")
    if arch not in ARCHITECTURES:
        raise NotImplementedError(f"GGUF architecture {arch!r} is not supported "
                                  f"(supported: {', '.join(ARCHITECTURES)})")
    loader = importlib.import_module(ARCHITECTURES[arch] + ".loader")
    return loader.load_engine(gguf_path, metadata, tensors, pack_dir, cache_gb, **options)
