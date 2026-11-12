"""Dequantizes ggml tensors with torch. Each function takes the raw bytes as a uint8 tensor of shape
[n_blocks, bytes_per_block] and returns float32 values of shape [n_blocks, elements_per_block].
Ported from the numpy reference in llama.cpp's gguf-py (gguf/quants.py)."""
import functools
import math

import torch
import torch.nn.functional as F

from .reader import BLOCK_SIZES, QK_K

def _f16(raw: torch.Tensor) -> torch.Tensor:
    """[n, 2] bytes -> [n, 1] float32"""
    return raw.contiguous().view(torch.float16).float()

def _shifts(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.uint8)

def _nibbles(qs: torch.Tensor) -> torch.Tensor:
    """[n, 16] packed bytes -> [n, 32]: the low nibbles are elements 0..15, the high nibbles 16..31."""
    return torch.cat((qs & 0x0F, qs >> 4), dim=-1)

def _high_bits(qh: torch.Tensor) -> torch.Tensor:
    """[n, 4] bytes (a little-endian uint32) -> [n, 32]: bit i of the uint32 for element i."""
    return ((qh[:, :, None] >> _shifts(*range(8))) & 1).reshape(qh.shape[0], 32)

def dequant_q4_0(b):
    d = _f16(b[:, :2])
    return d * (_nibbles(b[:, 2:]).float() - 8)

def dequant_q4_1(b):
    d, m = _f16(b[:, :2]), _f16(b[:, 2:4])
    return d * _nibbles(b[:, 4:]).float() + m

def dequant_q5_0(b):
    d = _f16(b[:, :2])
    q = _nibbles(b[:, 6:]) | (_high_bits(b[:, 2:6]) << 4)
    return d * (q.float() - 16)

def dequant_q5_1(b):
    d, m = _f16(b[:, :2]), _f16(b[:, 2:4])
    q = _nibbles(b[:, 8:]) | (_high_bits(b[:, 4:8]) << 4)
    return d * q.float() + m

def dequant_q8_0(b):
    d = _f16(b[:, :2])
    return d * b[:, 2:].contiguous().view(torch.int8).float()

def dequant_q2_k(b):
    n = b.shape[0]
    scales, qs = b[:, :16], b[:, 16:80]
    d, dmin = _f16(b[:, 80:82]), _f16(b[:, 82:84])
    dl = (d * (scales & 0x0F).float()).reshape(n, 16, 1)
    ml = (dmin * (scales >> 4).float()).reshape(n, 16, 1)
    q = (qs.reshape(n, 2, 1, 32) >> _shifts(0, 2, 4, 6).reshape(1, 1, 4, 1)) & 3
    return (dl * q.reshape(n, 16, 16).float() - ml).reshape(n, QK_K)

def dequant_q3_k(b):
    n = b.shape[0]
    hmask, qs, scales = b[:, :32], b[:, 32:96], b[:, 96:108]
    d = _f16(b[:, 108:110])
    # 16 6-bit scales: low 4 bits in bytes 0..7 (two per byte), high 2 bits in bytes 8..11 (four per byte)
    low = (scales[:, :8].reshape(n, 1, 8) >> _shifts(0, 4).reshape(1, 2, 1)).reshape(n, 16)
    high = (scales[:, 8:].reshape(n, 1, 4) >> _shifts(0, 2, 4, 6).reshape(1, 4, 1)).reshape(n, 16)
    sc = ((low & 0x0F) | ((high & 3) << 4)).float() - 32
    dl = (d * sc).reshape(n, 16, 1)
    ql = ((qs.reshape(n, 2, 1, 32) >> _shifts(0, 2, 4, 6).reshape(1, 1, 4, 1)) & 3).reshape(n, 16, 16)
    qh = ((hmask.reshape(n, 1, 1, 32) >> _shifts(*range(8)).reshape(1, 1, 8, 1)) & 1).reshape(n, 16, 16)
    # A set hmask bit means no offset; a clear one subtracts 4
    q = ql.float() - ((qh ^ 1) << 2).float()
    return (dl * q).reshape(n, QK_K)

def _k_scale_min(scales: torch.Tensor):
    """Q4_K/Q5_K: 12 bytes -> 8 6-bit scales and 8 6-bit mins."""
    n = scales.shape[0]
    s = scales.reshape(n, 3, 4)
    d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
    sc = torch.cat((d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)), dim=-1)
    mn = torch.cat((m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)), dim=-1)
    return sc.float(), mn.float()

def _q4_k_parts(b):
    """Q4_K blocks -> (scales [n, 8], mins [n, 8], 4-bit values [n, 8, 32]); sub-block j is scale * q - min."""
    n = b.shape[0]
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    sc, mn = _k_scale_min(b[:, 4:16])
    q = ((b[:, 16:].reshape(n, 4, 1, 32) >> _shifts(0, 4).reshape(1, 1, 2, 1)) & 0x0F).reshape(n, 8, 32)
    return d * sc, dmin * mn, q

def dequant_q4_k(b):
    scale, mn, q = _q4_k_parts(b)
    return (scale[..., None] * q.float() - mn[..., None]).reshape(b.shape[0], QK_K)

def dequant_q5_k(b):
    n = b.shape[0]
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    sc, mn = _k_scale_min(b[:, 4:16])
    d, dm = (d * sc).reshape(n, 8, 1), (dmin * mn).reshape(n, 8, 1)
    qh, qs = b[:, 16:48], b[:, 48:]
    ql = ((qs.reshape(n, 4, 1, 32) >> _shifts(0, 4).reshape(1, 1, 2, 1)) & 0x0F).reshape(n, 8, 32)
    qh = ((qh.reshape(n, 1, 1, 32) >> _shifts(*range(8)).reshape(1, 1, 8, 1)) & 1).reshape(n, 8, 32)
    return (d * (ql | (qh << 4)).float() - dm).reshape(n, QK_K)

def _q6_k_parts(b):
    """Q6_K blocks -> (scales [n, 16], signed 6-bit values [n, 16, 16] as int8); sub-block j is scale * q."""
    n = b.shape[0]
    ql, qh = b[:, :128], b[:, 128:192]
    scales = _f16(b[:, 208:210]) * b[:, 192:208].contiguous().view(torch.int8).float()
    ql = ((ql.reshape(n, 2, 1, 64) >> _shifts(0, 4).reshape(1, 1, 2, 1)) & 0x0F).reshape(n, 8, 32)
    qh = ((qh.reshape(n, 2, 1, 32) >> _shifts(0, 2, 4, 6).reshape(1, 1, 4, 1)) & 3).reshape(n, 8, 32)
    q = (ql | (qh << 4)).to(torch.int8) - 32
    return scales, q.reshape(n, 16, 16)

def dequant_q6_k(b):
    scales, q = _q6_k_parts(b)
    return (scales[..., None] * q.float()).reshape(b.shape[0], QK_K)

# IQ4_NL's 16 non-uniform levels (ggml's kvalues_iq4nl); each 4-bit value is an index into this table
IQ4_NL_LEVELS = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                             dtype=torch.float32)

def dequant_iq4_nl(b):
    d = _f16(b[:, :2])
    return d * IQ4_NL_LEVELS[_nibbles(b[:, 2:]).long()]

def dequant_iq4_xs(b):
    # 8 sub-blocks of 32: each has a 6-bit scale (low 4 bits in scales_l, high 2 in scales_h), offset by 32
    n = b.shape[0]
    d = _f16(b[:, :2])
    scales_h = b[:, 2].int() | (b[:, 3].int() << 8)
    high = (scales_h[:, None] >> torch.arange(0, 16, 2, dtype=torch.int32)) & 0x03         # [n, 8]
    low = ((b[:, 4:8, None] >> _shifts(0, 4)) & 0x0F).reshape(n, 8).int()                  # [n, 8]
    dl = d * ((low | (high << 4)) - 32).float()                                             # [n, 8]
    q = _nibbles(b[:, 8:].reshape(n * 8, 16)).reshape(n, 8, 32)
    return (dl[:, :, None] * IQ4_NL_LEVELS[q.long()]).reshape(n, QK_K)

@functools.cache
def _iq_grid(name: str) -> torch.Tensor:
    from . import iq_grids
    shape = getattr(iq_grids, name + "_SHAPE")
    raw = bytes.fromhex("".join(getattr(iq_grids, name)))
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).float().view(shape)

@functools.cache
def _ksigns_bits() -> torch.Tensor:
    """[128, 8] of +1/-1: ggml's ksigns_iq2xs (7 sign bits, the 8th making the parity even), one value per bit."""
    k = torch.tensor([i | ((bin(i).count("1") & 1) << 7) for i in range(128)], dtype=torch.uint8)
    return 1 - 2 * ((k[:, None] >> _shifts(*range(8))) & 1).float()

def _sign_bits(bytes_: torch.Tensor) -> torch.Tensor:
    """[n, m] bytes -> [n, m * 8] of +1/-1, bit i of each byte for its i-th value."""
    return 1 - 2 * ((bytes_[:, :, None] >> _shifts(*range(8))) & 1).reshape(bytes_.shape[0], -1).float()

def dequant_iq3_xxs(b):
    # 64 grid indices (4 values each), then 8 uint32: per 32 values, a 4-bit scale (top bits) and four 7-bit
    # indices into ksigns (one per 8 values)
    n = b.shape[0]
    d = _f16(b[:, :2])
    grid = _iq_grid("IQ3_XXS_GRID")[b[:, 2:66].long()].reshape(n, 8, 32)
    aux = b[:, 66:98].reshape(n, 8, 4).long()
    aux = aux[..., 0] | (aux[..., 1] << 8) | (aux[..., 2] << 16) | (aux[..., 3] << 24)    # [n, 8] uint32 values
    db = d * (0.5 + (aux >> 28).float()) * 0.5                                              # [n, 8]
    sign_idx = (aux[:, :, None] >> torch.tensor([0, 7, 14, 21])) & 0x7F                     # [n, 8, 4]
    signs = _ksigns_bits()[sign_idx].reshape(n, 8, 32)
    return (db[:, :, None] * grid * signs).reshape(n, QK_K)

def dequant_iq2_s(b):
    # 32 grid indices (8 values each; low 8 bits in qs, high 2 in qh), 32 bytes of explicit signs, then one 4-bit
    # scale per 16 values
    n = b.shape[0]
    d = _f16(b[:, :2])
    qh = ((b[:, 66:74, None] >> _shifts(0, 2, 4, 6)) & 0x03).reshape(n, 32).long()
    grid = _iq_grid("IQ2_S_GRID")[b[:, 2:34].long() | (qh << 8)].reshape(n, 16, 16)
    signs = _sign_bits(b[:, 34:66]).reshape(n, 16, 16)
    scales = ((b[:, 74:82, None] >> _shifts(0, 4)) & 0x0F).reshape(n, 16).float()
    db = d * (0.5 + scales) * 0.25                                                          # [n, 16]
    return (db[:, :, None] * grid * signs).reshape(n, QK_K)

def dequant_q2_0(b):
    # Value j is bits 2 * (j % 4) of byte j // 4 (ggml's dequantize_row_q2_0): 0, 1, 2, 3 -> -1, 0, +1, +2
    d = _f16(b[:, :2])
    q = (b[:, 2:, None] >> _shifts(0, 2, 4, 6)) & 0x03
    return d * (q.reshape(b.shape[0], -1).float() - 1)

DEQUANTIZERS = {
    "Q4_0": dequant_q4_0, "Q4_1": dequant_q4_1, "Q5_0": dequant_q5_0, "Q5_1": dequant_q5_1, "Q8_0": dequant_q8_0,
    "Q2_K": dequant_q2_k, "Q3_K": dequant_q3_k, "Q4_K": dequant_q4_k, "Q5_K": dequant_q5_k, "Q6_K": dequant_q6_k,
    "IQ4_NL": dequant_iq4_nl, "IQ4_XS": dequant_iq4_xs, "IQ3_XXS": dequant_iq3_xxs, "IQ2_S": dequant_iq2_s,
    "Q2_0": dequant_q2_0,
}
FLOAT_TYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
SUPPORTED_TYPES = set(DEQUANTIZERS) | set(FLOAT_TYPES)
CHUNK_ELEMENTS = 1 << 24

def dequantize(raw, type_name: str, shape: tuple, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """raw: a uint8 tensor (or anything torch.frombuffer accepts) holding exactly one tensor's bytes.
    Returns a new tensor of the given shape and dtype; it never shares memory with raw."""
    if not isinstance(raw, torch.Tensor):
        raw = torch.frombuffer(raw, dtype=torch.uint8)
    if type_name in FLOAT_TYPES:
        return raw.view(FLOAT_TYPES[type_name]).reshape(shape).to(dtype, copy=True)
    if type_name not in DEQUANTIZERS:
        raise NotImplementedError(f"ggml type {type_name} is not supported yet "
                                  f"(supported: {', '.join(sorted(SUPPORTED_TYPES))})")
    elements, block_bytes = BLOCK_SIZES[type_name]
    blocks = raw.reshape(-1, block_bytes)
    assert blocks.shape[0] * elements == math.prod(shape), (type_name, shape, raw.shape)
    # Chunks of 16M values keep the float32 temporaries small for big tensors like the embedding table
    chunk = CHUNK_ELEMENTS // elements
    if blocks.shape[0] <= chunk:
        return DEQUANTIZERS[type_name](blocks).to(dtype).reshape(shape)
    out = torch.empty(blocks.shape[0], elements, dtype=dtype)
    for start in range(0, blocks.shape[0], chunk):
        out[start:start + chunk] = DEQUANTIZERS[type_name](blocks[start:start + chunk])
    return out.reshape(shape)

class QuantizedEmbedding:
    """An embedding table kept as its ggml blocks in RAM, in place of a bf16 tensor. table[ids] (ids: a CPU tensor)
    dequantizes only those rows, giving the same bf16 values dequantize() gives for the whole table, as a CPU tensor.
    A Q4_K 248320 x 2048 table takes 0.27 GiB this way, against 0.95 GiB in bf16."""

    def __init__(self, raw: bytearray, type_name: str, shape: tuple):
        rows, cols = shape
        elements, block_bytes = BLOCK_SIZES[type_name]
        self.blocks = torch.frombuffer(raw, dtype=torch.uint8).view(rows, cols // elements * block_bytes)
        self.type_name, self.shape = type_name, tuple(shape)

    @staticmethod
    def supports(type_name: str, shape: tuple) -> bool:
        return type_name in DEQUANTIZERS and len(shape) == 2 and shape[1] % BLOCK_SIZES[type_name][0] == 0

    def to(self, device):
        return self  # the table stays in CPU memory; the engine copies the rows it looks up to its device

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        rows = self.blocks[ids]
        return dequantize(rows.reshape(-1), self.type_name, (rows.shape[0], self.shape[1]))

# Expert layouts the pack stores in place of the ggml blocks, so reading an expert costs little or no CPU:
# dequantizing ggml blocks with torch takes several times longer than reading them from the SSD.
#   INT4_G32 (from Q4_K and Q6_K): torch's CPU int4 matmul format. It is never dequantized; Int4Weight multiplies
#     with it. Bytes: packed 4-bit values [rows, k/2] uint8, then (scale, zero) per group of 32 as [k/32, rows, 2]
#     bf16. Q4_K converts exactly; Q6_K is requantized to 4 bits (some precision is lost).
#   INT8_G16 (older packs, from Q6_K): int8 values [rows, k], then one bf16 scale per 16 values [rows, k/16].
#     Dequantizing it is one multiply. Still readable, but no longer written.
#   INT4_G32_GPU (`pack --gpu`): the same, but packed in torch's MPS int4 format ([rows/8, k/(16 * tiles), 32,
#     tiles/2] int32), so a read only has to be copied to the GPU.
INT4_GROUP, INT8_GROUP = 32, 16
GPU = torch.device("mps")
MPS_INNER_K_TILES = 4  # tiling of the MPS int4 format (torch allows 2, 4 or 8)

def _to_bytes(t: torch.Tensor) -> bytes:
    out = bytearray(t.numel() * t.element_size())
    torch.frombuffer(out, dtype=torch.uint8).copy_(t.contiguous().reshape(-1).view(torch.uint8))
    return bytes(out)

def _blocks(raw, type_name: str) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, BLOCK_SIZES[type_name][1])

def pack_int4(q: torch.Tensor, scales_and_zeros: torch.Tensor, gpu: bool = False) -> bytes:
    """4-bit values [rows, k] (0..15) and (scale, zero) [k/32, rows, 2] -> INT4_G32 bytes (INT4_G32_GPU with gpu,
    packed by torch on the GPU, so it needs MPS)."""
    if gpu:
        pairs = ((q[:, 0::2] << 4) | q[:, 1::2]).to(torch.uint8)  # the MPS packer takes [rows, k/2], even k high
        packed = torch._convert_weight_to_int4pack(pairs.to(GPU), MPS_INNER_K_TILES).cpu()
    else:
        packed = torch._convert_weight_to_int4pack_for_cpu(q.to(torch.int32).contiguous(), 1)
    return _to_bytes(packed) + _to_bytes(scales_and_zeros.to(torch.bfloat16))

def q4_k_int4_parts(raw, shape: tuple):
    """Q4_K computes scale * q - min per sub-block of 32; the int4 kernel computes (q - 8) * scale + zero per
    group of 32, so zero = 8 * scale - min and the 4-bit values carry over unchanged (no precision is lost).
    Returns (q [rows, k] int32, scales_and_zeros [k/32, rows, 2] float32)."""
    rows, k = shape
    scale, mn, q = _q4_k_parts(_blocks(raw, "Q4_K"))
    scales_and_zeros = torch.stack((scale, 8 * scale - mn), dim=-1).reshape(rows, k // INT4_GROUP, 2).transpose(0, 1)
    return q.reshape(rows, k).to(torch.int32), scales_and_zeros

def float_to_int4_parts(w: torch.Tensor):
    """Quantizes a [rows, k] float weight to 4 bits, asymmetric min/max per group of 32 along k.
    Returns (q [rows, k] int32, scales_and_zeros [k/32, rows, 2] float32) in the INT4_G32 convention."""
    rows, k = w.shape
    g = w.float().reshape(rows, k // INT4_GROUP, INT4_GROUP)
    lo, hi = g.amin(dim=-1), g.amax(dim=-1)
    # The kernel reads scale and zero as bf16, so round them first and pick q against the rounded values
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16).float()
    zero = (lo + 8 * scale).to(torch.bfloat16).float()
    q = torch.round((g - zero[..., None]) / scale[..., None] + 8).clamp_(0, 15).to(torch.int32)
    return q.reshape(rows, k), torch.stack((scale, zero), dim=-1).transpose(0, 1)

def q4_k_to_int4(raw: bytes, shape: tuple, gpu: bool = False) -> bytes:
    return pack_int4(*q4_k_int4_parts(raw, shape), gpu)

def q6_k_to_int4(raw: bytes, shape: tuple, gpu: bool = False) -> bytes:
    """Dequantizes Q6_K at pack time and requantizes it to INT4_G32, so every expert runs on the int4 kernel
    and can be read straight into its cache slot."""
    rows, k = shape
    return pack_int4(*float_to_int4_parts(dequant_q6_k(_blocks(raw, "Q6_K")).reshape(rows, k)), gpu)

def q6_k_to_int8(raw: bytes, shape: tuple) -> bytes:
    rows, k = shape
    scales, q = _q6_k_parts(_blocks(raw, "Q6_K"))
    return _to_bytes(q) + _to_bytes(scales.to(torch.bfloat16))

def int4_nbytes(shape: tuple) -> int:
    rows, k = shape
    return rows * k // 2 + k // INT4_GROUP * rows * 2 * 2

def int8_nbytes(shape: tuple) -> int:
    rows, k = shape
    return rows * k + rows * k // INT8_GROUP * 2

# ggml type -> (pack layout, its size for a [rows, k] part, converter from the ggml bytes)
PACK_LAYOUTS = {"Q4_K": ("INT4_G32", int4_nbytes, q4_k_to_int4), "Q6_K": ("INT4_G32", int4_nbytes, q6_k_to_int4)}
PACK_LAYOUTS_GPU = {"Q4_K": ("INT4_G32_GPU", int4_nbytes, functools.partial(q4_k_to_int4, gpu=True)),
                    "Q6_K": ("INT4_G32_GPU", int4_nbytes, functools.partial(q6_k_to_int4, gpu=True))}
INT4_LAYOUTS = {"INT4_G32", "INT4_G32_GPU"}
INT4_ROW_MULTIPLE = 16  # torch's int4 packing needs the row count to be a multiple of 16

def int4_packable(shape, gpu: bool) -> bool:
    rows, k = shape
    return rows % INT4_ROW_MULTIPLE == 0 and (not gpu or k % (16 * MPS_INNER_K_TILES) == 0)

_staged, _staged_bytes = [], 0  # CPU tensors being copied to the GPU, kept alive until it has run the copies
STAGED_LIMIT = 256 * 1024**2

def release_staged():
    """Waits for the GPU (if anything is staged) and lets go of the CPU side of finished copies."""
    global _staged_bytes
    if _staged:
        torch.mps.synchronize()
        _staged.clear()
        _staged_bytes = 0

class Int4Weight:
    """A [rows, k] int4 weight, used in place of a bf16 tensor. F.linear(x, weight) runs torch's int4 kernel
    (through __torch_function__), so the model code needs no changes.

    An INT4_G32 weight computes on the CPU. An INT4_G32_GPU one (gpu=True) computes on the GPU (MPS): the first
    time it is used, or when to_gpu() is called, it is copied there without waiting, and its CPU bytes are let go
    once the copy has run (see release_staged). Call those on the main thread only. .device is where it computes,
    so callers can send their input there; the output comes back on the input's device."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        bias = (kwargs or {}).get("bias", args[2] if len(args) > 2 else None)
        if func is F.linear and len(args) > 1 and isinstance(args[1], Int4Weight) and bias is None:
            return args[1].linear(args[0])
        return NotImplemented

    def __init__(self, packed: torch.Tensor, scales_and_zeros: torch.Tensor, gpu: bool = False):
        self.packed, self.scales_and_zeros = packed, scales_and_zeros
        self.nbytes = packed.nbytes + scales_and_zeros.nbytes
        self.device = GPU if gpu else torch.device("cpu")

    @classmethod
    def from_buffer(cls, buf, offset: int, shape: tuple, gpu: bool = False) -> "Int4Weight":
        """Views the weight inside buf (no copy)."""
        rows, k = shape
        if gpu:
            packed = torch.frombuffer(buf, dtype=torch.int32, count=rows * k // 8, offset=offset)
            packed = packed.view(rows // 8, k // (16 * MPS_INNER_K_TILES), 32, MPS_INNER_K_TILES // 2)
        else:
            packed = torch.frombuffer(buf, dtype=torch.uint8, count=rows * k // 2, offset=offset).view(rows, k // 2)
        scales_and_zeros = torch.frombuffer(buf, dtype=torch.bfloat16, count=k // INT4_GROUP * rows * 2,
                                            offset=offset + rows * k // 2)
        return cls(packed, scales_and_zeros.view(k // INT4_GROUP, rows, 2), gpu)

    @classmethod
    def from_gpu_bytes(cls, raw: torch.Tensor, shape: tuple) -> "Int4Weight":
        """Views an INT4_G32_GPU weight inside raw, a uint8 tensor already on the GPU (no copy)."""
        rows, k = shape
        packed = raw[:rows * k // 2].view(torch.int32)
        packed = packed.view(rows // 8, k // (16 * MPS_INNER_K_TILES), 32, MPS_INNER_K_TILES // 2)
        scales_and_zeros = raw[rows * k // 2:int4_nbytes(shape)].view(torch.bfloat16)
        return cls(packed, scales_and_zeros.view(k // INT4_GROUP, rows, 2), gpu=True)

    def to_gpu(self):
        global _staged_bytes
        if self.device != GPU or self.packed.device == GPU:
            return
        if _staged_bytes + self.nbytes > STAGED_LIMIT:
            release_staged()
        _staged.append((self.packed, self.scales_and_zeros))
        _staged_bytes += self.nbytes
        self.packed = self.packed.to(GPU, non_blocking=True)
        self.scales_and_zeros = self.scales_and_zeros.to(GPU, non_blocking=True)

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if self.device == GPU:
            self.to_gpu()
            y = torch._weight_int4pack_mm(flat.to(GPU), self.packed, INT4_GROUP, self.scales_and_zeros).to(x.device)
        else:
            y = torch._weight_int4pack_mm_for_cpu(flat, self.packed, INT4_GROUP, self.scales_and_zeros)
        return y.view(*x.shape[:-1], y.shape[-1])

def dequant_int8_g16(buf, offset: int, shape: tuple) -> torch.Tensor:
    """An INT8_G16 weight inside buf -> a new bf16 tensor."""
    rows, k = shape
    q = torch.frombuffer(buf, dtype=torch.int8, count=rows * k, offset=offset).view(rows, k // INT8_GROUP, INT8_GROUP)
    scales = torch.frombuffer(buf, dtype=torch.bfloat16, count=rows * k // INT8_GROUP, offset=offset + rows * k)
    return (q.to(torch.bfloat16) * scales.view(rows, k // INT8_GROUP, 1)).view(rows, k)
