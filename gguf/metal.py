"""Metal kernels that compute straight from ggml's quantized blocks on the GPU, so experts run in the GGUF's own
formats: exactly the published weights, and as small as the GGUF stores them. Compiled at runtime with
torch.mps.compile_shader (no build step).

For each supported type, a device function decodes one 32-value chunk of a row (the same math as quants.py's
dequantizers, which match gguf-py exactly). Two kernels use them:
- gemv_<type>: y = W x for up to MAX_TOKENS tokens, bf16 in and out (summed in float32), so a matmul is one
  kernel launch. One SIMD group (32 threads) per output row: each thread decodes every 32nd chunk of the row
  once and multiplies it with every token, then the group sums.
- dequant_<type>: W -> bf16, for bigger batches (prefill), which then run as a bf16 matmul.

GGMLWeight wraps a tensor's raw bytes so F.linear(x, weight) runs them (like quants.Int4Weight)."""
import functools

import torch
import torch.nn.functional as F

from .reader import BLOCK_SIZES

MAX_TOKENS = 8  # gemv handles up to this many tokens; more are dequantized to bf16 and multiplied by torch
ROWS_PER_GROUP = 4  # SIMD groups (rows) per threadgroup
CHUNK = 32

DEQUANT_FUNCTIONS = r"""
inline float f16_at(device const uchar* p) { return float(as_type<half>(ushort(p[0] | (p[1] << 8)))); }
inline uint u32_at(device const uchar* p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

// Q5_0: blocks of 32, 22 bytes: d (f16), the 5th bits (u32), then 16 bytes of low nibbles
inline void deq_Q5_0(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + c * 22;
    float d = f16_at(b);
    uint qh = u32_at(b + 2);
    for (int j = 0; j < 16; j++) {
        uchar q = b[6 + j];
        v[j] = d * (float((q & 0xF) | (((qh >> j) & 1) << 4)) - 16.0f);
        v[j + 16] = d * (float((q >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16.0f);
    }
}

// IQ4_NL: blocks of 32, 18 bytes: d, then 16 bytes of 4-bit indices into kvalues_iq4nl
inline void deq_IQ4_NL(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + c * 18;
    float d = f16_at(b);
    for (int j = 0; j < 16; j++) {
        uchar q = b[2 + j];
        v[j] = d * (kv[q & 0xF]);
        v[j + 16] = d * (kv[q >> 4]);
    }
}

// IQ4_XS: blocks of 256, 136 bytes: d, scales_h (u16), scales_l[4], then 8 x 16 bytes of indices. Chunk c is
// sub-block c % 8 of block c / 8, with a 6-bit scale offset by 32
inline void deq_IQ4_XS(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + (c / 8) * 136;
    uint ib = c % 8;
    float d = f16_at(b);
    uint scales_h = uint(b[2]) | (uint(b[3]) << 8);
    uint ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    float dl = d * (float(ls) - 32.0f);
    device const uchar* qs = b + 8 + 16 * ib;
    for (int j = 0; j < 16; j++) {
        v[j] = dl * (kv[qs[j] & 0xF]);
        v[j + 16] = dl * (kv[qs[j] >> 4]);
    }
}

// IQ3_XXS: blocks of 256, 98 bytes: d, 64 grid indices (4 values each), then 8 u32: per 32 values a 4-bit scale
// (top bits) and four 7-bit indices into ksigns (8 values each)
inline void deq_IQ3_XXS(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + (c / 8) * 98;
    uint ib = c % 8;
    float d = f16_at(b);
    uint aux = u32_at(b + 66 + 4 * ib);
    float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    device const uchar* qs = b + 2 + 8 * ib;
    for (int l = 0; l < 4; l++) {
        uint s = (aux >> (7 * l)) & 127;
        uint signs = s | ((popcount(s) & 1) << 7);
        uint g1 = iq3xxs_grid[qs[2 * l]], g2 = iq3xxs_grid[qs[2 * l + 1]];
        for (int j = 0; j < 4; j++) {
            v[8 * l + j] = db * float((g1 >> (8 * j)) & 0xFF) * ((signs >> j) & 1 ? -1.0f : 1.0f);
            v[8 * l + j + 4] = db * float((g2 >> (8 * j)) & 0xFF) * ((signs >> (j + 4)) & 1 ? -1.0f : 1.0f);
        }
    }
}

// IQ2_S: blocks of 256, 82 bytes: d, 32 grid index low bytes, 32 sign bytes, 8 bytes of index high bits (2 each),
// then 8 bytes of 4-bit scales (one per 16 values)
inline void deq_IQ2_S(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + (c / 8) * 82;
    uint ib = c % 8;
    float d = f16_at(b);
    for (int l = 0; l < 4; l++) {
        uint g = 4 * ib + l;  // grid index within the block
        uint index = b[2 + g] | (((b[66 + g / 4] >> (2 * (g % 4))) & 3) << 8);
        ulong grid = iq2s_grid[index];
        uint m = 2 * ib + l / 2;  // scale index: one per 16 values
        float dl = d * (0.5f + float((b[74 + m / 2] >> (4 * (m % 2))) & 0xF)) * 0.25f;
        uchar signs = b[34 + g];
        for (int j = 0; j < 8; j++) {
            v[8 * l + j] = dl * float((grid >> (8 * j)) & 0xFF) * ((signs >> j) & 1 ? -1.0f : 1.0f);
        }
    }
}

// Q4_K: blocks of 256, 144 bytes: d, dmin, 12 bytes of 6-bit scales and mins (8 each), then 4 x 32 bytes of
// nibbles (low nibbles: sub-block 2g, high: 2g + 1). Value: d * scale * q - dmin * min
inline void deq_Q4_K(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + (c / 8) * 144;
    uint j = c % 8;
    float d = f16_at(b), dmin = f16_at(b + 2);
    device const uchar* s = b + 4;
    uint sc, mn;
    if (j < 4) {
        sc = s[j] & 63;
        mn = s[j + 4] & 63;
    } else {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 2) & 0x30);
        mn = (s[j + 4] >> 4) | ((s[j] >> 2) & 0x30);
    }
    float scale = d * float(sc), m = dmin * float(mn);
    device const uchar* qs = b + 16 + 32 * (j / 2);
    uint shift = 4 * (j % 2);
    for (int l = 0; l < 32; l++) v[l] = scale * float((qs[l] >> shift) & 0xF) - m;
}

// Q6_K: blocks of 256, 210 bytes: ql[128] (low 4 bits), qh[64] (high 2 bits), 16 int8 scales (one per 16
// values), then d. Chunk c is sub-block c % 8: half h = sub / 4 of the block, quarter s = sub % 4 of the half
inline void deq_Q6_K(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + (c / 8) * 210;
    uint sub = c % 8, h = sub / 4, s = sub % 4;
    float d = f16_at(b + 208);
    device const uchar* ql = b + 64 * h + 32 * (s % 2);
    device const uchar* qh = b + 128 + 32 * h;
    uint shift = 4 * (s / 2);
    float scale0 = d * float(as_type<char>(b[192 + 2 * sub])), scale1 = d * float(as_type<char>(b[193 + 2 * sub]));
    for (int l = 0; l < 32; l++) {
        int q = int(((ql[l] >> shift) & 0xF) | (((qh[l] >> (2 * s)) & 3) << 4)) - 32;
        v[l] = (l < 16 ? scale0 : scale1) * float(q);
    }
}

// Q2_0: blocks of 64, 18 bytes: d, then 2 bits per value (bits 2 * (j % 4) of byte j / 4), 0..3 -> -1..+2
inline void deq_Q2_0(device const uchar* row, uint c, thread float* v, threadgroup const float* kv) {
    device const uchar* b = row + (c / 2) * 18;
    float d = f16_at(b);
    uint first = 32 * (c % 2);
    for (int j = 0; j < 32; j++) {
        uint k = first + j;
        v[j] = d * (float((b[2 + k / 4] >> (2 * (k % 4))) & 3) - 1.0f);
    }
}
"""

# IQ4's 16 levels, copied into threadgroup memory first: neighbouring threads look up different entries, which is
# slow from constant memory (as llama.cpp's Metal kernels do)
LOAD_KV = r"""threadgroup float kv[16];
    if (tid < 16) kv[tid] = float(kvalues_iq4nl[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);"""

KERNEL_TEMPLATE = r"""
kernel void gemv_NAME(device bfloat* y [[buffer(0)]], device const uchar* w [[buffer(1)]],
                      device const bfloat* x [[buffer(2)]], constant uint& n_rows [[buffer(3)]],
                      constant uint& k [[buffer(4)]], constant uint& row_bytes [[buffer(5)]],
                      constant uint& tokens [[buffer(6)]],
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                      uint tid [[thread_index_in_threadgroup]]) {
    LOAD_KV
    uint row = gid / 32;
    if (row >= n_rows) return;  // whole SIMD groups return together
    device const uchar* r = w + row * row_bytes;
    float acc[MAX_TOKENS];
    for (uint t = 0; t < tokens; t++) acc[t] = 0.0f;
    float v[32];
    for (uint c = lane; c < k / 32; c += 32) {
        deq_NAME(r, c, v, kv);
        for (uint t = 0; t < tokens; t++) {
            device const bfloat* xt = x + t * k + c * 32;
            float s = 0.0f;
            for (int j = 0; j < 32; j++) s += v[j] * float(xt[j]);
            acc[t] += s;
        }
    }
    for (uint t = 0; t < tokens; t++) {
        float s = simd_sum(acc[t]);
        if (lane == 0) y[t * n_rows + row] = bfloat(s);
    }
}

kernel void dequant_NAME(device bfloat* out [[buffer(0)]], device const uchar* w [[buffer(1)]],
                         constant uint& n_rows [[buffer(2)]], constant uint& k [[buffer(3)]],
                         constant uint& row_bytes [[buffer(4)]], uint gid [[thread_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    LOAD_KV
    uint chunks = k / 32, row = gid / chunks, c = gid % chunks;
    if (row >= n_rows) return;
    float v[32];
    deq_NAME(w + row * row_bytes, c, v, kv);
    device bfloat* o = out + row * k + c * 32;
    for (int j = 0; j < 32; j++) o[j] = bfloat(v[j]);
}
"""

TYPES = ("Q5_0", "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ2_S", "Q2_0", "Q4_K", "Q6_K")  # the types these kernels run

def _tables() -> str:
    """ggml's lookup tables as Metal constants."""
    from . import iq_grids
    from .quants import IQ4_NL_LEVELS
    iq3 = bytes.fromhex("".join(iq_grids.IQ3_XXS_GRID))
    iq2 = bytes.fromhex("".join(iq_grids.IQ2_S_GRID))
    u32 = [int.from_bytes(iq3[i:i + 4], "little") for i in range(0, len(iq3), 4)]
    u64 = [int.from_bytes(iq2[i:i + 8], "little") for i in range(0, len(iq2), 8)]
    return (f"constant char kvalues_iq4nl[16] = {{{', '.join(str(int(v)) for v in IQ4_NL_LEVELS)}}};\n"
            f"constant uint iq3xxs_grid[{len(u32)}] = {{{', '.join(f'{v:#x}u' for v in u32)}}};\n"
            f"constant ulong iq2s_grid[{len(u64)}] = {{{', '.join(f'{v:#x}ul' for v in u64)}}};\n")

@functools.cache
def library():
    source = ("#include <metal_stdlib>\nusing namespace metal;\n"
              f"#define MAX_TOKENS {MAX_TOKENS}\n" + _tables() + DEQUANT_FUNCTIONS
              + "".join(KERNEL_TEMPLATE.replace("NAME", name).replace("LOAD_KV", LOAD_KV) for name in TYPES))
    return torch.mps.compile_shader(source)

def available() -> bool:
    return torch.backends.mps.is_available()

def row_bytes(type_name: str, k: int) -> int:
    elements, block_bytes = BLOCK_SIZES[type_name]
    return k // elements * block_bytes

def gemv(raw: torch.Tensor, type_name: str, shape: tuple, x: torch.Tensor) -> torch.Tensor:
    """raw: the weight's bytes (uint8, on the GPU); x: bf16 [tokens <= MAX_TOKENS, k] -> bf16 [tokens, rows]."""
    rows, k = shape
    x = x.to(torch.bfloat16).contiguous()
    y = torch.empty(x.shape[0], rows, dtype=torch.bfloat16, device=raw.device)
    getattr(library(), "gemv_" + type_name)(y, raw, x, rows, k, row_bytes(type_name, k), x.shape[0],
                                            threads=rows * 32, group_size=32 * ROWS_PER_GROUP)
    return y

def dequantize_gpu(raw: torch.Tensor, type_name: str, shape: tuple) -> torch.Tensor:
    """raw (uint8, on the GPU) -> bf16 [rows, k] on the GPU."""
    rows, k = shape
    out = torch.empty(rows, k, dtype=torch.bfloat16, device=raw.device)
    getattr(library(), "dequant_" + type_name)(out, raw, rows, k, row_bytes(type_name, k),
                                               threads=rows * (k // CHUNK), group_size=256)
    return out

class GGMLWeight:
    """A [rows, k] weight kept as its ggml blocks on the GPU, used in place of a bf16 tensor: F.linear(x, weight)
    runs the Metal kernels above (through __torch_function__). raw is a uint8 tensor already on the GPU (a view of
    an expert slot), or on the CPU, in which case it's copied there on first use without waiting (see
    quants.Int4Weight.to_gpu, whose staging it shares)."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        bias = (kwargs or {}).get("bias", args[2] if len(args) > 2 else None)
        if func is F.linear and len(args) > 1 and isinstance(args[1], GGMLWeight) and bias is None:
            return args[1].linear(args[0])
        return NotImplemented

    def __init__(self, raw: torch.Tensor, type_name: str, shape: tuple):
        self.raw, self.type_name, self.shape = raw, type_name, tuple(shape)
        self.nbytes = raw.numel()
        self.device = torch.device("mps")

    def to_gpu(self):
        from . import quants
        if self.raw.device.type == "mps":
            return
        if quants._staged_bytes + self.nbytes > quants.STAGED_LIMIT:
            quants.release_staged()
        quants._staged.append((self.raw,))
        quants._staged_bytes += self.nbytes
        self.raw = self.raw.to(self.device, non_blocking=True)

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        self.to_gpu()
        flat = x.reshape(-1, x.shape[-1]).to(self.device)
        if flat.shape[0] <= MAX_TOKENS:
            y = gemv(self.raw, self.type_name, self.shape, flat).to(x.dtype)
        else:
            y = F.linear(flat.to(torch.bfloat16), dequantize_gpu(self.raw, self.type_name, self.shape)).to(x.dtype)
        return y.view(*x.shape[:-1], y.shape[-1]).to(x.device)
