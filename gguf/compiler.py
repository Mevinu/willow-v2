import functools
import json
import os
import re
import sys

from safetensor.compiler import PAGE_ALIGNMENT, PART_ORDER, align_up
from .reader import read_gguf, model_stem

# llama.cpp stores each layer's experts stacked in one tensor: blk.N.ffn_{gate,up,down}_exps.weight
# (or a fused ffn_gate_up_exps), PyTorch shape [num_experts, out, in], quantized along the last dim
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$")

def default_pack_dir(gguf_path: str) -> str:
    return model_stem(gguf_path) + ".expert_pack"

def num_decoder_layers(metadata: dict) -> int:
    """Layers the engine runs: block_count minus any multi-token-prediction layers appended after them."""
    arch = metadata["general.architecture"]
    return metadata[f"{arch}.block_count"] - metadata.get(f"{arch}.nextn_predict_layers", 0)

def scan_experts(metadata: dict, tensors: dict):
    """Works out where every expert's bytes live in the GGUF file(s), from the header alone.
    Returns the same (num_layers, num_experts, largest expert payload, plan, part_info) as
    safetensor.compiler.scan_experts, with ggml type names (e.g. "Q4_K") as the part dtypes."""
    num_layers = num_decoder_layers(metadata)
    plan = {}
    part_info = {}
    for name, t in tensors.items():
        m = EXPERT_RE.match(name)
        if not m or int(m.group(1)) >= num_layers:
            continue
        layer, part = int(m.group(1)), m.group(2)
        # Quant types differ between layers (e.g. Q4_K_M uses Q6_K for some down_exps), so shape and type are
        # recorded per layer, under "<layer>.<part>"
        part_info[f"{layer}.{part}"] = {"shape": list(t.shape[1:]), "dtype": t.type}
        # Expert e is the e-th equal slice: quant blocks never cross rows, so the slices split cleanly
        n = t.shape[0]
        size = t.nbytes // n
        for e in range(n):
            plan.setdefault((layer, e), []).append((part, t.path, t.offset + e * size, size))

    if not plan:
        sys.exit("No MoE expert tensors found (expected names like blk.N.ffn_gate_exps.weight)")

    layers = sorted({l for l, _ in plan})
    num_experts = max(e for _, e in plan) + 1
    if layers != list(range(num_layers)):
        sys.exit(f"Expected experts in layers 0..{num_layers - 1}, found {layers[0]}..{layers[-1]} "
                 f"({len(layers)} layers); models with dense layers are not supported yet")
    for key in [(l, e) for l in layers for e in range(num_experts)]:
        parts = sorted(plan.get(key, []), key=lambda p: PART_ORDER.index(p[0]))
        names = [p[0] for p in parts]
        if names not in (["gate", "up", "down"], ["gate_up", "down"]):
            sys.exit(f"Expert {key[0]}_{key[1]} has parts {names}, expected gate, up and down (or gate_up and down)")
        plan[key] = parts

    max_payload = max(sum(p[3] for p in parts) for parts in plan.values())
    return num_layers, num_experts, max_payload, plan, part_info

def compile_pack(gguf_path: str, out_dir: str = None):
    """Copies a GGUF model's expert tensors, still quantized, into experts.pack + manifest.json
    (default: <model>.expert_pack/ next to the .gguf file). Q4_K and Q6_K parts are converted to layouts that
    are cheap to run (see quants.PACK_LAYOUTS); other types are copied byte for byte."""
    from .quants import PACK_LAYOUTS  # needs torch, which only GGUF packs need (for the conversions)

    out_dir = out_dir or default_pack_dir(gguf_path)
    os.makedirs(out_dir, exist_ok=True)

    metadata, tensors = read_gguf(gguf_path)
    num_layers, num_experts, _, plan, part_info = scan_experts(metadata, tensors)
    types = sorted({info["dtype"] for info in part_info.values()})

    converters, sizes = {}, {}
    for key, info in part_info.items():
        if info["dtype"] in PACK_LAYOUTS:
            layout, nbytes, convert = PACK_LAYOUTS[info["dtype"]]
            converters[key] = functools.partial(convert, shape=tuple(info["shape"]))
            sizes[key] = nbytes(info["shape"])
            part_info[key] = {"shape": info["shape"], "dtype": layout, "source_dtype": info["dtype"]}
    max_payload = max(sum(sizes.get(f"{layer}.{p[0]}", p[3]) for p in parts) for (layer, _), parts in plan.items())
    layouts = sorted({info["dtype"] for info in part_info.values()} - set(types))
    print(f"Found {num_layers} layers x {num_experts} experts, largest expert = {max_payload / 1024**2:.2f} MiB, "
          f"types = {', '.join(types)}" + (f" (stored as {', '.join(layouts)})" if layouts else ""))

    pack_path = os.path.join(out_dir, "experts.pack")
    manifest_path = os.path.join(out_dir, "manifest.json")
    build_converted_pack(plan, part_info, converters, pack_path, manifest_path, num_layers, num_experts, max_payload)
    print(f"Wrote {pack_path} ({os.path.getsize(pack_path) / 1024**3:.2f} GiB) and {manifest_path}")

def build_converted_pack(plan: dict, part_info: dict, converters: dict, pack_path: str, manifest_path: str,
                         num_layers: int, num_experts: int, max_payload: int):
    """Writes the same pack layout as safetensor.compiler.build_expert_pack (one 4KB-aligned block per expert),
    but passes each part's bytes through converters["<layer>.<part>"] when there is one."""
    import torch

    stride = align_up(max_payload, PAGE_ALIGNMENT)
    manifest = {
        "format": "gguf",
        "alignment": PAGE_ALIGNMENT,
        "data_start": 0,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_stride": stride,
        "parts": part_info,
        "experts": {},
        # The INT4_G32 layout comes from a private torch function, so the engine checks it runs the same version
        "torch_version": torch.__version__,
    }
    fds = {}
    try:
        with open(pack_path, "wb") as pack_file:
            for layer in range(num_layers):
                for expert in range(num_experts):
                    parts, payload = {}, bytearray()
                    for part, path, offset, size in plan[(layer, expert)]:
                        if path not in fds:
                            fds[path] = os.open(path, os.O_RDONLY)
                        data = os.pread(fds[path], size, offset)
                        if len(data) != size:
                            raise IOError(f"Short read for {part} of expert {layer}_{expert} in {path}")
                        convert = converters.get(f"{layer}.{part}")
                        data = convert(data) if convert else data
                        parts[part] = {"offset": len(payload), "size": len(data)}
                        payload += data
                    if len(payload) > stride:
                        raise ValueError(f"Expert payload ({len(payload)}B) exceeds aligned stride ({stride}B)")
                    file_offset = (layer * num_experts + expert) * stride
                    assert pack_file.tell() == file_offset
                    pack_file.write(payload + bytes(stride - len(payload)))
                    manifest["experts"][f"{layer}_{expert}"] = {
                        "file_offset": file_offset,
                        "read_length": align_up(len(payload), PAGE_ALIGNMENT),
                        "parts": parts,
                    }
                print(f"  layer {layer + 1}/{num_layers} packed", flush=True)
    finally:
        for fd in fds.values():
            os.close(fd)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
