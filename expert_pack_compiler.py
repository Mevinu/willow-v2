import os
import json
import math

PAGE_ALIGNMENT = 4096  # Mandatory 4KB alignment for Linux O_DIRECT

def align_up(size: int, alignment: int = PAGE_ALIGNMENT) -> int:
    return int(math.ceil(size / alignment) * alignment)

def build_expert_pack(
    expert_plan: dict,
    part_info: dict,
    output_pack_path: str,
    output_manifest_path: str,
    num_layers: int,
    num_experts: int,
    expert_stride: int
):
    """
    Reorganizes scattered MoE expert tensors into a single contiguous,
    4096-byte page-aligned binary pack file for O_DIRECT streaming.

    expert_plan maps (layer, expert) -> [(part_name, shard_path, byte_offset, byte_size), ...]
    giving the raw bytes of each part (e.g. gate/up/down) inside the source safetensors shards.
    part_info maps part_name -> {"shape": [...], "dtype": "BF16"} and is copied into the manifest.
    """
    expert_stride = align_up(expert_stride, PAGE_ALIGNMENT)
    manifest = {
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
