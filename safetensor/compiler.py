import os
import re
import sys
import json
import math
import struct

PAGE_ALIGNMENT = 4096  # Mandatory 4KB alignment for Linux O_DIRECT

# Matches both expert layouts:
#   one tensor per expert:  model.layers.N.mlp.experts.M.gate_proj.weight      (Qwen3-MoE, OLMoE)
#   all experts stacked:    model.language_model.layers.N.mlp.experts.gate_up_proj  [experts, ...]  (Qwen3.5-MoE)
EXPERT_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\."
    r"(?:(\d+)\.(gate_proj|up_proj|down_proj)\.weight|(gate_up_proj|down_proj))$"
)
PART_NAMES = {"gate_proj": "gate", "up_proj": "up", "gate_up_proj": "gate_up", "down_proj": "down"}
PART_ORDER = ["gate", "up", "gate_up", "down"]

def align_up(size: int, alignment: int = PAGE_ALIGNMENT) -> int:
    return int(math.ceil(size / alignment) * alignment)

def read_safetensors_header(path: str, with_data_start: bool = False):
    # safetensors layout: 8-byte little-endian header length, then a JSON header, then tensor data
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return (header, 8 + header_len) if with_data_start else header

def build_index_map(model_dir: str) -> dict:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return {name: os.path.join(model_dir, shard) for name, shard in weight_map.items()}

    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_path):
        return {name: single_path for name in read_safetensors_header(single_path) if name != "__metadata__"}

    sys.exit(f"No model.safetensors.index.json or model.safetensors found in {model_dir}")

def scan_experts(index_map: dict):
    """Reads only the safetensors headers and works out where every expert's bytes live.

    Returns (num_layers, num_experts, largest expert payload, plan, part_info), where plan maps
    (layer, expert) -> [(part, shard, byte_offset, byte_size)] in PART_ORDER.
    """
    plan = {}
    part_info = {}

    def add_part(layer, expert, part, shard, offset, size, shape, dtype):
        info = {"shape": shape, "dtype": dtype}
        if part_info.setdefault(part, info) != info:
            sys.exit(f"Expert part {part!r} has inconsistent shape/dtype: {part_info[part]} vs {info}")
        plan.setdefault((layer, expert), []).append((part, shard, offset, size))

    for shard in sorted(set(index_map.values())):
        header, data_start = read_safetensors_header(shard, with_data_start=True)
        for name, info in header.items():
            m = EXPERT_RE.match(name)
            if not m:
                continue
            layer = int(m.group(1))
            start, end = info["data_offsets"]
            if m.group(2) is not None:
                part = PART_NAMES[m.group(3)]
                add_part(layer, int(m.group(2)), part, shard, data_start + start, end - start, info["shape"], info["dtype"])
            else:
                # Stacked [num_experts, ...]: expert e is the e-th equal slice of the tensor's bytes
                part = PART_NAMES[m.group(4)]
                n = info["shape"][0]
                size = (end - start) // n
                for e in range(n):
                    add_part(layer, e, part, shard, data_start + start + e * size, size, info["shape"][1:], info["dtype"])

    if not plan:
        sys.exit("No MoE expert tensors found (expected names like "
                 "model.layers.N.mlp.experts.M.gate_proj.weight or model.language_model.layers.N.mlp.experts.gate_up_proj)")

    layers = sorted({l for l, _ in plan})
    num_experts = max(e for _, e in plan) + 1
    if layers != list(range(len(layers))):
        sys.exit(f"Expert layers are not contiguous from 0 (found {layers[0]}..{layers[-1]}); "
                 "models with leading dense layers are not supported yet")

    expected_parts = sorted(part_info, key=PART_ORDER.index)
    for key in [(l, e) for l in layers for e in range(num_experts)]:
        parts = sorted(plan.get(key, []), key=lambda p: PART_ORDER.index(p[0]))
        if [p[0] for p in parts] != expected_parts:
            sys.exit(f"Expert {key[0]}_{key[1]} has parts {[p[0] for p in parts]}, expected {expected_parts}")
        plan[key] = parts

    max_payload = max(sum(p[3] for p in parts) for parts in plan.values())
    return len(layers), num_experts, max_payload, plan, part_info

def compile_pack(model_dir: str, out_dir: str = None):
    """Finds a model's expert tensors and writes experts.pack + manifest.json (default: <model_dir>/expert_pack)."""
    out_dir = out_dir or os.path.join(model_dir, "expert_pack")
    os.makedirs(out_dir, exist_ok=True)

    index_map = build_index_map(model_dir)
    num_layers, num_experts, max_payload, plan, part_info = scan_experts(index_map)
    parts = ", ".join(f"{name} {info['dtype']}{info['shape']}" for name, info in part_info.items())
    print(f"Found {num_layers} layers x {num_experts} experts, largest expert = {max_payload / 1024**2:.2f} MiB, parts = {parts}")

    pack_path = os.path.join(out_dir, "experts.pack")
    manifest_path = os.path.join(out_dir, "manifest.json")
    build_expert_pack(plan, part_info, pack_path, manifest_path, num_layers, num_experts, max_payload)
    print(f"Wrote {pack_path} ({os.path.getsize(pack_path) / 1024**3:.2f} GiB) and {manifest_path}")

def build_expert_pack(
    expert_plan: dict,
    part_info: dict,
    output_pack_path: str,
    output_manifest_path: str,
    num_layers: int,
    num_experts: int,
    expert_stride: int,
    pack_format: str = "safetensors"
):
    """
    Reorganizes scattered MoE expert tensors into a single contiguous,
    4096-byte page-aligned binary pack file for O_DIRECT streaming.

    expert_plan maps (layer, expert) -> [(part_name, shard_path, byte_offset, byte_size), ...]
    giving the raw bytes of each part (e.g. gate/up/down) inside the source safetensors shards.
    part_info maps part_name -> {"shape": [...], "dtype": "BF16"} and is copied into the manifest.
    pack_format records where the bytes came from ("safetensors" or "gguf"), so engines can refuse the other kind.
    """
    expert_stride = align_up(expert_stride, PAGE_ALIGNMENT)
    manifest = {
        "format": pack_format,
        "alignment": PAGE_ALIGNMENT,
        "data_start": 0,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_stride": expert_stride,
        "parts": part_info,
        "experts": {}
    }

    shard_fds = {}
    try:
        with open(output_pack_path, "wb") as pack_file:
            for layer_idx in range(num_layers):
                for expert_idx in range(num_experts):
                    # 1. Read each part's raw bytes straight from the shards (dtype-agnostic)
                    parts = {}
                    raw_payload = b""
                    for part_name, shard_path, byte_offset, byte_size in expert_plan[(layer_idx, expert_idx)]:
                        if shard_path not in shard_fds:
                            shard_fds[shard_path] = os.open(shard_path, os.O_RDONLY)
                        data = os.pread(shard_fds[shard_path], byte_size, byte_offset)
                        if len(data) != byte_size:
                            raise IOError(f"Short read for {part_name} of expert {layer_idx}_{expert_idx} in {shard_path}")

                        # 2. Concatenate back-to-back: [gate | up | down] (or [gate_up | down])
                        parts[part_name] = {"offset": len(raw_payload), "size": byte_size}
                        raw_payload += data

                    payload_len = len(raw_payload)
                    if payload_len > expert_stride:
                        raise ValueError(f"Expert payload ({payload_len}B) exceeds aligned stride ({expert_stride}B)")

                    # 3. Pad with zeros to exact 4096-byte stride boundary
                    padding_len = expert_stride - payload_len
                    aligned_block = raw_payload + (b"\x00" * padding_len)

                    # 4. Compute deterministic O(1) offset and write block
                    expected_offset = (layer_idx * num_experts + expert_idx) * expert_stride
                    assert pack_file.tell() == expected_offset
                    pack_file.write(aligned_block)

                    # 5. Record internal tensor boundaries within the fetched block
                    manifest["experts"][f"{layer_idx}_{expert_idx}"] = {
                        "file_offset": expected_offset,
                        "read_length": align_up(payload_len, PAGE_ALIGNMENT),
                        "parts": parts
                    }
                print(f"  layer {layer_idx + 1}/{num_layers} packed", flush=True)
    finally:
        for fd in shard_fds.values():
            os.close(fd)

    with open(output_manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
