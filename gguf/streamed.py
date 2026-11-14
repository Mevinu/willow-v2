"""Embedding tables too big for RAM (Qwen3.8's 27 GiB per-layer n-gram table), read row by row from the GGUF
file itself. Nothing is copied out of the GGUF and nothing stays resident: each lookup reads only the rows it
needs, and the OS file cache keeps recently read pages in spare RAM."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from .quants import DEQUANTIZERS, dequantize
from .reader import BLOCK_SIZES, GGUFTensor

PARALLEL_ROWS = 64  # lookups with more unique rows than this are read on a thread pool (a prefill batch)

class StreamedEmbedding:
    """A [rows, cols] GGUF tensor left on disk. table[ids] (a CPU tensor of row ids, any shape) reads and
    dequantizes just those rows, returning bf16 [*ids.shape, cols] on the CPU, the same values dequantize() gives
    for the whole table. Row r is at tensor.offset + r * row_bytes, since ggml blocks never cross rows."""

    def __init__(self, tensor: GGUFTensor, read_threads: int = 8):
        if len(tensor.shape) != 2 or tensor.type not in DEQUANTIZERS:
            raise NotImplementedError(f"{tensor.name}: can't stream a {tensor.type} tensor of shape {tensor.shape}")
        rows, cols = tensor.shape
        elements, block_bytes = BLOCK_SIZES[tensor.type]
        if cols % elements:
            raise ValueError(f"{tensor.name}: rows of {cols} don't split into {tensor.type} blocks of {elements}")
        self.tensor, self.shape = tensor, tuple(tensor.shape)
        self.row_bytes = cols // elements * block_bytes
        assert self.row_bytes * rows == tensor.nbytes, (tensor.name, self.row_bytes, tensor.nbytes)
        self.fd = os.open(tensor.path, os.O_RDONLY)
        self.pool = ThreadPoolExecutor(read_threads)
        self.lock = threading.Lock()
        self.reads = 0
        self.bytes_read = 0

    def to(self, device):
        return self  # rows are read on the CPU; the engine copies them to its device

    def read_row(self, row: int, out: memoryview):
        got = os.preadv(self.fd, [out], self.tensor.offset + row * self.row_bytes)
        if got != self.row_bytes:
            raise IOError(f"short read for row {row} of {self.tensor.name}: {got} of {self.row_bytes} bytes")

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(ids)
        unique, inverse = torch.unique(ids.flatten(), return_inverse=True)
        rows = unique.tolist()
        if rows and (rows[0] < 0 or rows[-1] >= self.shape[0]):
            raise IndexError(f"row ids must be in 0..{self.shape[0] - 1}, got {rows[0]}..{rows[-1]}")
        raw = bytearray(len(rows) * self.row_bytes)
        with memoryview(raw) as view:
            slots = [view[i * self.row_bytes:(i + 1) * self.row_bytes] for i in range(len(rows))]
            if len(rows) > PARALLEL_ROWS:
                list(self.pool.map(self.read_row, rows, slots))  # list() surfaces any read error
            else:
                for row, slot in zip(rows, slots):
                    self.read_row(row, slot)
            del slots
        with self.lock:
            self.reads += len(rows)
            self.bytes_read += len(raw)
        values = dequantize(raw, self.tensor.type, (len(rows), self.shape[1]))
        return values[inverse].reshape(*ids.shape, self.shape[1])

    def close(self):
        self.pool.shutdown()
        os.close(self.fd)
