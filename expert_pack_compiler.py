import os
import json
import math
import torch
from safetensors import safe_open

PAGE_ALIGNMENT = 4096  # Mandatory 4KB alignment for Linux O_DIRECT

def align_up(size: int, alignment: int = PAGE_ALIGNMENT) -> int:
    return int(math.ceil(size / alignment) * alignment)

def build_expert_pack(
    safetensors_index_Map: dict,
    output_pack_path: str,
    output_manifest_path: str,
    num_layers: int,
    num_experts: int,
    expert_stride: int
):
    """
    Reorganizes scattered MoE expert tensors (gate, up, down) into a single
    contiguous, 4096-byte page-aligned binary pack file for O_DIRECT streaming.
    """
    expert_stride = align_up(expert_stride, PAGE_ALIGNMENT)
    manifest = {
        "alignment": PAGE_ALIGNMENT,
        "data_start": 0,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_stride": expert_stride,
        "experts": {}
    }

    with open(output_pack_path, "wb") as pack_file:
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                # 1. Load the three scattered projections for (layer_idx, expert_idx)
                gate_bytes = load_quantized_tensor_bytes(safetensors_index_Map, layer_idx, expert_idx, "gate_proj")
                up_bytes   = load_quantized_tensor_bytes(safetensors_index_Map, layer_idx, expert_idx, "up_proj")
                down_bytes = load_quantized_tensor_bytes(safetensors_index_Map, layer_idx, expert_idx, "down_proj")

                # 2. Concatenate back-to-back: [gate | up | down]
                raw_payload = gate_bytes + up_bytes + down_bytes
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
                    "gate_offset": 0,
                    "gate_size": len(gate_bytes),
                    "up_offset": len(gate_bytes),
                    "up_size": len(up_bytes),
                    "down_offset": len(gate_bytes) + len(up_bytes),
                    "down_size": len(down_bytes)
                }

    with open(output_manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

def load_quantized_tensor_bytes(index_map: dict, layer: int, expert: int, proj: str) -> bytes:
    tensor_name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
    shard_file = index_map[tensor_name]
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(tensor_name)
        return bytes(tensor.untyped_storage())