import os
import re
import sys
import mmap
import struct
from dataclasses import dataclass

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32  # used when general.alignment is not set

# ggml type id -> (name, elements per block, bytes per block). Packing works for every type here (it copies
# bytes); running needs a dequantizer in quants.py.
QK_K = 256
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 2 + 16),
    3: ("Q4_1", 32, 2 + 2 + 16),
    6: ("Q5_0", 32, 2 + 4 + 16),
    7: ("Q5_1", 32, 2 + 2 + 4 + 16),
    8: ("Q8_0", 32, 2 + 32),
    9: ("Q8_1", 32, 2 + 2 + 32),
    10: ("Q2_K", QK_K, 2 + 2 + QK_K // 16 + QK_K // 4),
    11: ("Q3_K", QK_K, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: ("Q4_K", QK_K, 2 + 2 + QK_K // 2 + 12),
    13: ("Q5_K", QK_K, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: ("Q6_K", QK_K, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    15: ("Q8_K", QK_K, 4 + QK_K + QK_K // 8),
    16: ("IQ2_XXS", QK_K, 2 + QK_K // 4),
    17: ("IQ2_XS", QK_K, 2 + QK_K // 4 + QK_K // 32),
    18: ("IQ3_XXS", QK_K, 2 + QK_K // 4 + QK_K // 8),
    19: ("IQ1_S", QK_K, 2 + QK_K // 8 + QK_K // 16),
    20: ("IQ4_NL", 32, 2 + 16),
    21: ("IQ3_S", QK_K, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    22: ("IQ2_S", QK_K, 2 + QK_K // 4 + QK_K // 16),
    23: ("IQ4_XS", QK_K, 2 + 2 + QK_K // 2 + QK_K // 64),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", QK_K, QK_K // 8 + QK_K // 16 + QK_K // 32),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", QK_K, 2 + 4 * 13),
    35: ("TQ2_0", QK_K, 2 + 64),
    39: ("MXFP4", 32, 1 + 16),
}
BLOCK_SIZES = {name: (elements, size) for name, elements, size in GGML_TYPES.values()}

# Metadata value types: id -> struct format (strings and arrays are handled separately)
SCALAR_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9

# Split models: <name>-00001-of-00003.gguf
SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")

@dataclass
class GGUFTensor:
    name: str
    path: str          # file holding the tensor (differs per split)
    offset: int        # absolute byte offset of its data in that file
    type: str          # ggml type name, e.g. "Q4_K"
    shape: tuple       # PyTorch order (outermost first): the reverse of GGUF's ne
    nbytes: int

def tensor_nbytes(shape: tuple, type_name: str) -> int:
    elements, size = BLOCK_SIZES[type_name]
    if shape[-1] % elements:
        raise ValueError(f"row length {shape[-1]} is not a multiple of the {type_name} block size {elements}")
    count = 1
    for d in shape:
        count *= d
    return count // elements * size

class _Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def unpack(self, fmt: str):
        values = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return values[0] if len(values) == 1 else values

    def string(self) -> str:
        length = self.unpack("Q")
        text = bytes(self.buf[self.pos:self.pos + length]).decode("utf-8", errors="replace")
        self.pos += length
        return text

    def value(self, value_type: int):
        if value_type == STRING:
            return self.string()
        if value_type == ARRAY:
            item_type, count = self.unpack("IQ")
            if item_type in SCALAR_FORMATS:
                # One unpack for the whole array: token-type and score arrays have ~250k entries
                fmt = f"{count}{SCALAR_FORMATS[item_type]}"
                values = list(struct.unpack_from("<" + fmt, self.buf, self.pos))
                self.pos += struct.calcsize("<" + fmt)
                return values
            return [self.value(item_type) for _ in range(count)]
        return self.unpack(SCALAR_FORMATS[value_type])

def read_gguf_file(path: str):
    """Parses one .gguf file's header. Returns (metadata dict, {name: GGUFTensor})."""
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        c = _Cursor(memoryview(mm))
        try:
            if bytes(c.buf[:4]) != GGUF_MAGIC:
                sys.exit(f"{path} is not a GGUF file")
            c.pos = 4
            version, tensor_count, kv_count = c.unpack("IQQ")
            if version < 2:
                sys.exit(f"{path} is GGUF version {version}; only versions 2 and 3 are supported")

            metadata = {}
            for _ in range(kv_count):
                key = c.string()
                metadata[key] = c.value(c.unpack("I"))

            infos = []
            for _ in range(tensor_count):
                name = c.string()
                n_dims = c.unpack("I")
                ne = c.unpack(f"{n_dims}Q") if n_dims > 1 else (c.unpack("Q"),)
                type_id, offset = c.unpack("IQ")
                if type_id not in GGML_TYPES:
                    sys.exit(f"{path}: tensor {name} has unknown ggml type {type_id}")
                infos.append((name, tuple(reversed(ne)), GGML_TYPES[type_id][0], offset))

            alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
            data_start = (c.pos + alignment - 1) // alignment * alignment
        finally:
            c.buf.release()

    tensors = {
        name: GGUFTensor(name, path, data_start + offset, type_name, shape, tensor_nbytes(shape, type_name))
        for name, shape, type_name, offset in infos
    }
    return metadata, tensors

def split_paths(path: str) -> list:
    """All files of a split model given any one of them, or [path] for a single-file model."""
    m = SPLIT_RE.match(os.path.basename(path))
    if not m:
        return [path]
    base, _, count = m.groups()
    folder = os.path.dirname(path)
    paths = [os.path.join(folder, f"{base}-{i:05d}-of-{count}.gguf") for i in range(1, int(count) + 1)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"Split GGUF is missing {len(missing)} of {count} files, e.g. {missing[0]}")
    return paths

def model_stem(path: str) -> str:
    """The model's path without .gguf or the split suffix: /m/Qwen-Q4_K_M-00001-of-00002.gguf -> /m/Qwen-Q4_K_M."""
    m = SPLIT_RE.match(os.path.basename(path))
    name = m.group(1) if m else os.path.basename(path).removesuffix(".gguf")
    return os.path.join(os.path.dirname(path), name)

def read_gguf(path: str):
    """Reads a GGUF model (all of its split files). Returns (metadata, {name: GGUFTensor}); metadata comes
    from the first file, which is the only one that holds the model's hyperparameters and tokenizer."""
    metadata, tensors = None, {}
    for p in split_paths(path):
        file_metadata, file_tensors = read_gguf_file(p)
        metadata = metadata if metadata is not None else file_metadata
        tensors.update(file_tensors)
    return metadata, tensors
