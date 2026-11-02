import os
import re
import sys
import json
import struct
import argparse

from expert_pack_compiler import build_expert_pack

# Matches both expert layouts:
#   one tensor per expert:  model.layers.N.mlp.experts.M.gate_proj.weight      (Qwen3-MoE, OLMoE)
#   all experts stacked:    model.language_model.layers.N.mlp.experts.gate_up_proj  [experts, ...]  (Qwen3.5-MoE)
EXPERT_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\."
    r"(?:(\d+)\.(gate_proj|up_proj|down_proj)\.weight|(gate_up_proj|down_proj))$"
)
PART_NAMES = {"gate_proj": "gate", "up_proj": "up", "gate_up_proj": "gate_up", "down_proj": "down"}
PART_ORDER = ["gate", "up", "gate_up", "down"]

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

def main():
    parser = argparse.ArgumentParser(description="Compile MoE expert weights into a 4KB-aligned pack file")
    parser.add_argument("model_dir", help="Hugging Face model folder containing .safetensors files")
    parser.add_argument("--out-dir", default=None, help="Where to write the pack (default: <model_dir>/expert_pack)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.model_dir, "expert_pack")
    os.makedirs(out_dir, exist_ok=True)

    index_map = build_index_map(args.model_dir)
    num_layers, num_experts, max_payload, plan, part_info = scan_experts(index_map)
    parts = ", ".join(f"{name} {info['dtype']}{info['shape']}" for name, info in part_info.items())
    print(f"Found {num_layers} layers x {num_experts} experts, largest expert = {max_payload / 1024**2:.2f} MiB, parts = {parts}")

    pack_path = os.path.join(out_dir, "experts.pack")
    manifest_path = os.path.join(out_dir, "manifest.json")
    build_expert_pack(plan, part_info, pack_path, manifest_path, num_layers, num_experts, max_payload)
    print(f"Wrote {pack_path} ({os.path.getsize(pack_path) / 1024**3:.2f} GiB) and {manifest_path}")

if __name__ == "__main__":
    main()
