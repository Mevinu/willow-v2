import functools
import json
import os
import re
import sys

from .reader import read_gguf, model_stem

# 16KB: the VM page size on Apple Silicon (and a multiple of Linux O_DIRECT's 4KB). With F_NOCACHE, reads that
# aren't aligned to whole 16KB pages go through a bounce buffer in the kernel.
PAGE_ALIGNMENT = 16384
PART_ORDER = ["gate", "up", "gate_up", "down"]

def align_up(size: int, alignment: int = PAGE_ALIGNMENT) -> int:
    return -(-size // alignment) * alignment

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
    Returns (num_layers, num_experts, largest expert payload in bytes, plan, part_info): plan maps
    (layer, expert) to its [(part, path, offset, size), ...] in PART_ORDER, and part_info maps "<layer>.<part>"
    to its {"shape", "dtype"}, with ggml type names (e.g. "Q4_K") as the dtypes."""
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

def fuse_gate_up(plan: dict, part_info: dict):
    """Turns each layer's separate gate and up parts into one gate_up part (gate rows, then up rows), so the
    engine runs one matmul for both. ggml quant blocks never cross rows, so appending up's bytes to gate's is
    a valid [2 * rows, k] tensor of the same type. Layers whose gate and up differ in type stay separate.
    Returns (plan, part_info), with plan's parts as (part, [(path, offset, size), ...]): sources read back to back."""
    fused_layers = set()
    for layer in {l for l, _ in plan}:
        gate, up = part_info.get(f"{layer}.gate"), part_info.get(f"{layer}.up")
        if gate and up and gate == up:
            fused_layers.add(layer)
            part_info[f"{layer}.gate_up"] = {"shape": [2 * gate["shape"][0], gate["shape"][1]], "dtype": gate["dtype"]}
            del part_info[f"{layer}.gate"], part_info[f"{layer}.up"]
    fused_plan = {}
    for (layer, expert), parts in plan.items():
        sources = {part: (path, offset, size) for part, path, offset, size in parts}
        if layer in fused_layers:
            fused_plan[(layer, expert)] = [("gate_up", [sources["gate"], sources["up"]]), ("down", [sources["down"]])]
        else:
            fused_plan[(layer, expert)] = [(part, [src]) for part, src in sources.items()]
    return fused_plan, part_info

def compile_pack(gguf_path: str, out_dir: str = None, gpu: bool = None):
    """Copies a GGUF model's expert tensors, still quantized, into experts.pack + manifest.json
    (default: <model>.expert_pack/ next to the .gguf file). gate and up are fused into one gate_up part, and
    With gpu, every part is copied byte for byte: gguf/metal.py runs Q4_K, Q6_K and Qwen3.8's types (IQ4_XS,
    IQ3_XXS, IQ2_S, IQ4_NL, Q5_0, Q2_0) as they are. Without it, those are converted to int4 for the CPU kernel
    (see quants.PACK_LAYOUTS: Q4_K exactly, the rest requantized); other types are copied byte for byte. gpu stores the int4 parts in the GPU (MPS) kernel's format, so they
    run on the GPU (default: when torch can use MPS)."""
    import torch  # only GGUF packs need it (for the conversions)
    from .quants import PACK_LAYOUTS, PACK_LAYOUTS_GPU, int4_packable

    if gpu is None:
        gpu = torch.backends.mps.is_available()
    elif gpu and not torch.backends.mps.is_available():
        raise ValueError("--gpu needs torch with MPS (Apple Silicon)")
    print(f"experts will run on the {'GPU' if gpu else 'CPU'}" + ("" if gpu else " (pack --gpu to run them on the GPU)"))

    out_dir = out_dir or default_pack_dir(gguf_path)
    os.makedirs(out_dir, exist_ok=True)

    metadata, tensors = read_gguf(gguf_path)
    num_layers, num_experts, _, plan, part_info = scan_experts(metadata, tensors)
    types = sorted({info["dtype"] for info in part_info.values()})

    plan, part_info = fuse_gate_up(plan, part_info)

    converters, sizes = {}, {}
    for key, info in part_info.items():
        layouts = PACK_LAYOUTS_GPU if gpu else PACK_LAYOUTS
        if info["dtype"] in layouts and int4_packable(info["shape"], gpu):
            layout, nbytes, convert = layouts[info["dtype"]]
            converters[key] = functools.partial(convert, shape=tuple(info["shape"]))
            sizes[key] = nbytes(info["shape"])
            part_info[key] = {"shape": info["shape"], "dtype": layout, "source_dtype": info["dtype"]}
    # Each layer's experts get slots the size of its largest one: quant types differ between layers (a 2-bit layer
    # next to a 4-bit one), and one model-wide size would pad every small expert up to the biggest
    layer_payloads = {}
    for (layer, _), parts in plan.items():
        payload = sum(sizes.get(f"{layer}.{part}", sum(src[2] for src in sources)) for part, sources in parts)
        layer_payloads[layer] = max(layer_payloads.get(layer, 0), payload)
    max_payload = max(layer_payloads.values())
    layouts = sorted({info["dtype"] for info in part_info.values()} - set(types))
    print(f"Found {num_layers} layers x {num_experts} experts, largest expert = {max_payload / 1024**2:.2f} MiB, "
          f"types = {', '.join(types)}" + (f" (stored as {', '.join(layouts)})" if layouts else ""))
    pack_path = os.path.join(out_dir, "experts.pack")
    manifest_path = os.path.join(out_dir, "manifest.json")
    build_converted_pack(plan, part_info, converters, pack_path, manifest_path, num_layers, num_experts,
                         layer_payloads)
    print(f"Wrote {pack_path} ({os.path.getsize(pack_path) / 1024**3:.2f} GiB) and {manifest_path}")

def build_converted_pack(plan: dict, part_info: dict, converters: dict, pack_path: str, manifest_path: str,
                         num_layers: int, num_experts: int, layer_payloads: dict):
    """Writes experts.pack (one page-aligned block per expert, layer after layer; a layer's blocks are all the size
    of its largest expert, layer_payloads[layer] rounded up to whole pages) and its manifest.json, which records
    where each expert is. Each part's bytes go through converters["<layer>.<part>"] when there is one. plan's parts
    are (part, [(path, offset, size), ...]), as fuse_gate_up returns them."""
    import torch

    strides = [align_up(layer_payloads[layer], PAGE_ALIGNMENT) for layer in range(num_layers)]
    manifest = {
        "format": "gguf",
        "alignment": PAGE_ALIGNMENT,
        "data_start": 0,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "layer_strides": strides,
        "parts": part_info,
        "experts": {},
        # The int4 layouts come from private torch functions, so the engine checks it runs the same version
        "torch_version": torch.__version__,
    }
    fds = {}
    try:
        with open(pack_path, "wb") as pack_file:
            file_offset = 0
            for layer in range(num_layers):
                stride = strides[layer]
                for expert in range(num_experts):
                    parts, payload = {}, bytearray()
                    for part, sources in plan[(layer, expert)]:
                        data = b""
                        for path, offset, size in sources:
                            if path not in fds:
                                fds[path] = os.open(path, os.O_RDONLY)
                            chunk = os.pread(fds[path], size, offset)
                            if len(chunk) != size:
                                raise IOError(f"Short read for {part} of expert {layer}_{expert} in {path}")
                            data += chunk
                        convert = converters.get(f"{layer}.{part}")
                        data = convert(data) if convert else data
                        parts[part] = {"offset": len(payload), "size": len(data)}
                        payload += data
                    if len(payload) > stride:
                        raise ValueError(f"Expert payload ({len(payload)}B) exceeds aligned stride ({stride}B)")
                    assert pack_file.tell() == file_offset
                    pack_file.write(payload + bytes(stride - len(payload)))
                    manifest["experts"][f"{layer}_{expert}"] = {
                        "file_offset": file_offset,
                        "read_length": align_up(len(payload), PAGE_ALIGNMENT),
                        "parts": parts,
                    }
                    file_offset += stride
                print(f"  layer {layer + 1}/{num_layers} packed", flush=True)
    finally:
        for fd in fds.values():
            os.close(fd)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
