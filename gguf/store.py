"""Streams routed experts from an expert pack (built by `main.py pack`, see compiler.py) into the engine. Nothing
here depends on the model's architecture: a pack is just (layer, expert) -> its gate/up/down parts."""
import fcntl
import json
import math
import mmap
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice

import torch

from .compiler import PAGE_ALIGNMENT, align_up
from .quants import (INT4_LAYOUTS, PACK_LAYOUTS, Int4Weight, dequant_int8_g16, dequantize, int4_nbytes,
                     release_staged)

USAGE_FILE = "expert_usage.json"  # the expert usage table, next to manifest.json in the pack
SLAB_CHUNK_BYTES = 1024**3  # largest single buffer a slab is allocated as
GPU_SLAB_MIN = 64           # slots a GPU pack reads into when the cache is off or smaller
PRELOAD_FRACTION = 0.15  # share of each layer's experts, most used first, read into the cache at load (plain LRU)
HOT_SHARE = 0.75  # share of a slab's slots kept for the hot set: the most used experts, by the usage table

def expert_weights(parts: dict) -> tuple:
    """An expert's parts by name -> (gate_up, down) when gate and up are fused into one
    [2 * intermediate, hidden] matrix (gate rows first), else (gate, up, down)."""
    if "gate_up" in parts:
        return parts["gate_up"], parts["down"]
    return parts["gate"], parts["up"], parts["down"]

class ExpertStore:
    """Reads one expert's [gate | up | down] (or [gate_up | down]) block from the pack with a single aligned read.
    A layer's experts are read in parallel on a thread pool (os.preadv releases the GIL), so the SSD has many
    requests in flight and later reads finish while earlier experts are being computed. INT4_G32 parts stay 4-bit
    (Int4Weight); the rest are dequantized to bf16 on the read threads.

    With cache_bytes, experts are kept in a slab: memory allocated once and cut into fixed slots, one expert each.
    Up to HOT_SHARE of the slots hold the hot set, the most used experts by the usage table (across all layers,
    so a layer whose routing is more concentrated gets more of them); preload reads them in. Every other read
    (misses, prefetches) goes into the rest of the slots, which are reused least recently used first; a hot
    expert is only evicted when none of those can be. Plain LRU over every slot doesn't work when a token reads
    more experts than the cache holds (Qwen3.8: ~500 per token): each token would evict the whole cache, the
    most used experts included, before anything is reused. Its size is exact (the weights live in the slots, and nothing
    else is kept alive), and the OS file cache is turned off for the pack so the same bytes aren't held twice.
    Layers whose slots are the same size share a slab, with the same number of slots per layer. A slot holds the
    expert as it is computed with: int4 parts as stored, everything else as bf16, so a hit costs nothing. Layers
    that are all int4 (every layer, in a pack of a Q4_K/Q6_K model) are read straight into their slot.
    Only the main thread changes the cache; a read thread just fills the one slot it was given. If the slabs can't
    be built (cache too small), a plain LRU of up to cache_bytes is used instead.

    A pack built with `pack --gpu` (INT4_G32_GPU parts) computes its experts on the GPU. Its slabs are memory the
    CPU and GPU share (see mps_shared), so an expert is read from the SSD straight into memory the GPU computes
    from, with no copy. They always exist (with --cache-gb 0, just enough slots to read into, in GPU_SLAB_MIN), and
    the GPU runs behind the CPU, so a slot handed out for the current layer isn't refilled until load_many starts
    the next layer: before that, the engine copies the next layer's expert choices back to the CPU, which waits
    for all queued GPU work."""

    # prefetch does nothing while this many reads are already running (None: never skip). An expert is small
    # (~2 MiB), so the guesses queue behind this layer's reads and finish while its experts and the next attention
    # compute; capped, they were skipped on exactly the layers with the most misses.
    prefetch_max_in_flight = None

    def __init__(self, pack_dir: str, cache_bytes: int = 0, read_threads: int = 8, read_ahead: int = 16):
        with open(os.path.join(pack_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        if "parts" not in self.manifest:
            raise ValueError(f"{pack_dir} was built by an older main.py; rerun `main.py pack` on the model to rebuild it")
        if self.manifest.get("format", "safetensors") != "gguf":
            raise ValueError(f"{pack_dir} is a {self.manifest.get('format', 'safetensors')} pack, which is no longer "
                             "supported; rerun `main.py pack` on the .gguf model")
        self.fd = os.open(os.path.join(pack_dir, "experts.pack"), os.O_RDONLY)
        # "TYPE42": Q2_0 in packs built before the reader knew its name (the bytes are the same)
        self.parts = {name: (tuple(info["shape"]), "Q2_0" if info["dtype"] == "TYPE42" else info["dtype"])
                      for name, info in self.manifest["parts"].items()}
        self.cache = OrderedDict()  # (layer, expert) -> weights, least recently used first (plain LRU only)
        self.cache_bytes = cache_bytes
        self.cached_bytes = 0
        self.hits = 0
        self.reads = 0
        self.bytes_read = 0
        self.read_seconds = 0.0  # wall time with at least one read in flight (not summed per thread)
        self.pool = ThreadPoolExecutor(read_threads)
        self.read_ahead = read_ahead  # max experts requested but not yet consumed (bounds memory during prefill)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.busy_since = 0.0
        self.busy = set()     # (layer, expert) handed out by load_many and not yet consumed
        self.prefetched = {}  # (layer, expert) -> (future, token): prefetch reads not yet picked up
        self.prefetches = 0       # prefetch reads started
        self.prefetch_used = 0    # of those, ones load_many then asked for

        types = {type_name for _, type_name in self.parts.values()}
        if types & INT4_LAYOUTS and self.manifest.get("torch_version") != torch.__version__:
            raise ValueError(f"{pack_dir} was built with torch {self.manifest.get('torch_version')}, and its int4 "
                             f"layout may not match torch {torch.__version__}; rerun `main.py pack` on this model")
        if types & set(PACK_LAYOUTS):
            print(f"note: {pack_dir} stores {', '.join(sorted(types & set(PACK_LAYOUTS)))} experts as raw ggml "
                  "blocks, which are slow to dequantize; rerun `main.py pack` on this model to convert them")
        if "INT8_G16" in types:
            print(f"note: {pack_dir} stores Q6_K experts as int8, which is dequantized as each expert is read; rerun "
                  "`main.py pack` on this model to store them as int4 (smaller, and run without dequantizing)")
        self.gpu = "INT4_G32_GPU" in types
        if self.gpu and not torch.backends.mps.is_available():
            raise ValueError(f"{pack_dir} was packed for the GPU (`pack --gpu`), but torch can't use MPS here; "
                             "rerun `main.py pack --no-gpu` on this model")
        self.requested_cache = cache_bytes
        self.in_use = set()  # (layer, expert) the GPU may still be computing with (GPU slabs only)
        self.slabs = {}  # layer -> its slab group (shared by layers with the same slot layout)
        self.hot = set()  # (layer, expert) the slabs keep for as long as they can (see choose_hot)
        if cache_bytes or self.gpu:
            self.build_slabs(cache_bytes)
        self.load_usage(pack_dir)
        if self.slabs:
            self.choose_hot()

    # Loading a layer's experts

    def load_many(self, layer: int, experts: list):
        """Starts loading a layer's experts and returns an iterator of (expert, weights), where weights is
        (gate_up, down) or (gate, up, down). The reads for misses are submitted before this returns (up to
        read_ahead of them; for a decode step, all), so the SSD works on them while the caller computes the
        cache hits, which come first. Misses then come in the order their reads finish."""
        if self.gpu:
            # The last layer's output has been copied back, so the GPU is done with its slots and copies
            release_staged()
            self.in_use = {(layer, expert) for expert in experts}
        keys = [(layer, expert) for expert in experts]
        self.collect_prefetches(layer, exclude=set(keys))
        self.busy.update(keys)  # their cache entries must not be evicted until they have been consumed
        hits, misses = [], []
        for key in keys:
            weights = self.lookup(key)
            if weights is None and key in self.prefetched and self.prefetched[key][0].done():
                future, token = self.prefetched.pop(key)  # a prefetch that has already finished: as good as a hit
                weights = self.finish_read(key, future.result(), token)
                self.prefetch_used += 1
            if weights is not None:
                hits.append((key[1], weights))
            else:
                misses.append(key[1])

        reads = {}  # future -> (expert, token for finish_read)
        def start(expert):
            key = (layer, expert)
            if key in self.prefetched:  # prefetched, still reading
                self.prefetch_used += 1
                future, token = self.prefetched.pop(key)
            else:
                future, token = self.start_read(layer, expert)
            reads[future] = (expert, token)

        pending = iter(misses)
        for expert in islice(pending, self.read_ahead):
            start(expert)
        return self.results(layer, keys, hits, reads, pending, start)

    def results(self, layer: int, keys: list, hits: list, reads: dict, pending, start):
        try:
            for expert, weights in hits:
                yield expert, weights
                self.busy.discard((layer, expert))
            while reads:
                done, _ = wait(reads, return_when=FIRST_COMPLETED)
                for future in done:
                    expert, token = reads.pop(future)
                    weights = self.finish_read((layer, expert), future.result(), token)
                    following = next(pending, None)
                    if following is not None:
                        start(following)
                    yield expert, weights
                    self.busy.discard((layer, expert))
        finally:
            self.busy.difference_update(keys)

    def prefetch(self, layer: int, experts: list):
        """Starts reading experts that will probably be needed soon, without waiting for them. load_many reuses
        these reads (they are never read twice); ones it never asks for go into the cache once they finish, or
        are dropped if there is no cache. Skipped while the current layer still has prefetch_max_in_flight or
        more reads in flight, so guesses never hold up reads that are actually needed."""
        if self.prefetch_max_in_flight is not None and self.in_flight >= self.prefetch_max_in_flight:
            return
        for expert in experts:
            key = (layer, expert)
            if key in self.cache or key in self.prefetched:
                continue
            started = self.start_read(layer, expert, prefetch=True)
            if started is None:
                break
            self.prefetched[key] = started
            self.prefetches += 1

    def collect_prefetches(self, layer: int, exclude: set = frozenset()):
        """Deals with prefetches load_many isn't asking for, as it starts on layer: finished ones go into the cache.
        Without a cache, a guess is only worth anything to the layer it was made for, so guesses for this layer
        or earlier are dropped (cancelled, if their read hasn't started). Runs on the main thread."""
        for key, (future, token) in list(self.prefetched.items()):
            if key in exclude:
                continue
            if not self.cache_bytes and key[0] <= layer:
                del self.prefetched[key]
                future.cancel()
            elif future.done():
                del self.prefetched[key]
                self.finish_read(key, future.result(), token)

    # Cache hooks: with a slab, a read goes straight into a free slot; without one, into the plain LRU

    def lookup(self, key):
        """Returns a cached expert's weights (marking it most recently used), or None."""
        weights = self.cache.get(key)
        if weights is not None:
            if self.slabs:
                group = self.slabs[key[0]]
                (group["hot_lru"] if key in group["hot_lru"] else group["lru"]).move_to_end(key)
            else:
                self.cache.move_to_end(key)
            self.hits += 1
        return weights

    def start_read(self, layer: int, expert: int, prefetch: bool = False):
        """Submits a read to the pool. Returns (future, token passed to finish_read), or None to skip a prefetch."""
        if not self.slabs:
            return self.pool.submit(self.read, layer, expert), None
        group = self.slabs[layer]
        slot = self.take_slot(group)
        if slot is None:
            # Every slot is busy (a huge prefill batch): read without caching, unless it's only a guess
            return None if prefetch else (self.pool.submit(self.read, layer, expert), None)
        return self.pool.submit(self.fill, layer, expert, group, slot), slot

    def finish_read(self, key, weights, slot):
        """Called on the main thread once a read is done, with what the read returned. Returns the weights."""
        if not self.slabs:
            self.remember(key, weights)
            return weights
        if slot is not None:
            group = self.slabs[key[0]]
            if weights is None:  # a GPU slot: its views are made here, on the main thread
                weights = self.gpu_slot_weights(group, slot, key[0])
            (group["hot_lru"] if key in self.hot else group["lru"])[key] = slot
            self.cache[key] = weights
            self.cached_bytes += group["slot"]
        return weights

    def remember(self, key, weights):
        """Plain LRU (no slabs): keeps weights, evicting the least recently used to stay within cache_bytes."""
        size = sum(t.nbytes for t in weights)
        if size <= self.cache_bytes:
            while self.cached_bytes + size > self.cache_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= sum(t.nbytes for t in evicted)
            self.cache[key] = weights
            self.cached_bytes += size

    # Slabs

    def build_slabs(self, cache_bytes: int):
        num_experts = self.manifest["num_experts"]
        self.slot_parts = {}  # layer -> {part: (offset in slot, size in slot)}
        self.direct = {}      # layer -> whether its slot is a byte-for-byte copy of the pack block
        layouts = {}          # (direct, slot size) -> layers
        for layer in range(self.manifest["num_layers"]):
            entries = [self.manifest["experts"][f"{layer}_{e}"] for e in range(num_experts)]
            parts, offset = {}, 0
            for name in entries[0]["parts"]:
                shape, type_name = self.parts[f"{layer}.{name}"]
                size = int4_nbytes(shape) if type_name in INT4_LAYOUTS else math.prod(shape) * 2
                parts[name] = (offset, size)
                offset += size
            direct = all(self.parts[f"{layer}.{name}"][1] in INT4_LAYOUTS for name in parts)
            if direct:
                assert all(e["parts"][n]["offset"] == parts[n][0] for e in entries for n in parts)
                offset = max(e["read_length"] for e in entries)
            self.slot_parts[layer], self.direct[layer] = parts, direct
            layouts.setdefault((direct, align_up(offset, PAGE_ALIGNMENT)), []).append(layer)

        per_layer = cache_bytes // sum(slot * len(layers) for (_, slot), layers in layouts.items())
        fewest_layers = min(len(layers) for layers in layouts.values())
        if per_layer * fewest_layers < 2 * self.read_ahead:
            if not self.gpu:
                print("note: --cache-gb is too small to be useful; cache off")
                self.cache_bytes = 0
                return
            per_layer = max(per_layer, -(-GPU_SLAB_MIN // fewest_layers))
        self.slots_per_layer = per_layer
        if self.gpu:
            from .mps_shared import shared_empty
        for (_, slot), layers in layouts.items():
            count = per_layer * len(layers)
            # A slab is cut into chunks of whole slots (Metal limits the size of one buffer)
            per_chunk = max(1, min(count, SLAB_CHUNK_BYTES // slot))
            chunks = []
            for start in range(0, count, per_chunk):
                size = min(per_chunk, count - start) * slot
                chunks.append(shared_empty(size)[::-1] if self.gpu else (mmap.mmap(-1, size), None))
            group = {"chunks": chunks,  # (CPU-writable buffer, the same bytes as a GPU tensor or None)
                     "per_chunk": per_chunk, "slot": slot, "free": list(range(count)),
                     # (layer, expert) -> slot number, least recently used first: hot experts, and the rest
                     "hot_lru": OrderedDict(), "lru": OrderedDict(), "layers": layers}
            for layer in layers:
                self.slabs[layer] = group
        if sys.platform == "darwin":
            fcntl.fcntl(self.fd, getattr(fcntl, "F_NOCACHE", 48), 1)
        self.cache_bytes = sum(g["slot"] * len(g["free"]) for g in {id(g): g for g in self.slabs.values()}.values())

    def take_slot(self, group: dict):
        """A free slot, else the least recently used one not handed out by load_many, nor (GPU slabs) used since
        the GPU last caught up (None if there is none). Slots being filled are in neither the free list nor the LRU,
        so they are never taken."""
        if group["free"]:
            return group["free"].pop()
        for lru in (group["lru"], group["hot_lru"]):  # hot experts only when nothing else can go
            victim = next((key for key in lru if key not in self.busy and key not in self.in_use), None)
            if victim is not None:
                del self.cache[victim]
                self.cached_bytes -= group["slot"]
                return lru.pop(victim)
        return None

    def slot_place(self, group: dict, slot: int):
        """(CPU buffer, GPU tensor or None, offset of the slot in them)."""
        host, gpu = group["chunks"][slot // group["per_chunk"]]
        return host, gpu, slot % group["per_chunk"] * group["slot"]

    def gpu_slot_weights(self, group: dict, slot: int, layer: int):
        """An expert's weights as views of its GPU slot."""
        _, gpu, base = self.slot_place(group, slot)
        parts = {}
        for name, (offset, size) in self.slot_parts[layer].items():
            shape, type_name = self.parts[f"{layer}.{name}"]
            raw = gpu[base + offset:base + offset + size]
            parts[name] = (Int4Weight.from_gpu_bytes(raw, shape) if type_name in INT4_LAYOUTS
                           else raw.view(torch.bfloat16).view(shape))
        return expert_weights(parts)

    def slot_tensor(self, slab: mmap.mmap, base: int, layer: int, name: str):
        shape, type_name = self.parts[f"{layer}.{name}"]
        offset = base + self.slot_parts[layer][name][0]
        if type_name == "INT4_G32":
            return Int4Weight.from_buffer(slab, offset, shape)
        return torch.frombuffer(slab, dtype=torch.bfloat16, count=math.prod(shape), offset=offset).view(shape)

    # Reads (these run on the thread pool)

    def read(self, layer: int, expert: int):
        """Reads one expert from the pack into a new buffer (no slot)."""
        entry = self.manifest["experts"][f"{layer}_{expert}"]
        block = bytearray(entry["read_length"])
        self.pread(block, layer, expert, entry)
        return expert_weights({name: self.part_tensor(block, layer, name, part) for name, part in entry["parts"].items()})

    def fill(self, layer: int, expert: int, group: dict, slot: int):
        """Reads one expert into its slot and returns its weights, viewing the slot (None for a GPU slot: those
        views are made by finish_read on the main thread)."""
        entry = self.manifest["experts"][f"{layer}_{expert}"]
        slab, gpu, base = self.slot_place(group, slot)
        with memoryview(slab) as view:
            if self.direct[layer]:
                self.pread(view[base:base + entry["read_length"]], layer, expert, entry)
            else:
                block = bytearray(entry["read_length"])
                self.pread(block, layer, expert, entry)
                for name, part in entry["parts"].items():
                    offset, size = self.slot_parts[layer][name]
                    if self.parts[f"{layer}.{name}"][1] in INT4_LAYOUTS:
                        view[base + offset:base + offset + size] = memoryview(block)[part["offset"]:part["offset"] + size]
                    else:
                        self.slot_tensor(slab, base, layer, name).copy_(self.part_tensor(block, layer, name, part))
        if gpu is not None:
            return None
        return expert_weights({name: self.slot_tensor(slab, base, layer, name) for name in self.slot_parts[layer]})

    def pread(self, buf, layer: int, expert: int, entry: dict):
        """os.preadv straight into buf (os.pread would return bytes that then need copying), with I/O accounting."""
        with self.lock:
            if self.in_flight == 0:
                self.busy_since = time.perf_counter()
            self.in_flight += 1
        try:
            got = os.preadv(self.fd, [buf], entry["file_offset"])
        finally:
            with self.lock:
                self.in_flight -= 1
                if self.in_flight == 0:
                    self.read_seconds += time.perf_counter() - self.busy_since
        if got != len(buf):
            raise IOError(f"short read for expert {layer}_{expert}: {got} of {len(buf)} bytes")
        with self.lock:
            self.reads += 1
            self.bytes_read += len(buf)

    def part_tensor(self, block: bytearray, layer: int, name: str, part: dict):
        """Turns one part's bytes inside a read block into a weight: int4 parts view the block, others are
        dequantized to bf16."""
        shape, type_name = self.parts[f"{layer}.{name}"]
        if type_name in INT4_LAYOUTS:
            return Int4Weight.from_buffer(block, part["offset"], shape, gpu=type_name == "INT4_G32_GPU")
        if type_name == "INT8_G16":
            return dequant_int8_g16(block, part["offset"], shape)
        raw = torch.frombuffer(block, dtype=torch.uint8, count=part["size"], offset=part["offset"])
        return dequantize(raw, type_name, shape)

    # Expert usage table: how many tokens each (layer, expert) has been routed to, summed over every run on this
    # pack. Kept in RAM as a [num_layers, num_experts] count tensor, saved to USAGE_FILE in the pack after a run.

    def load_usage(self, pack_dir: str):
        self.usage_path = os.path.join(pack_dir, USAGE_FILE)
        shape = (self.manifest["num_layers"], self.manifest["num_experts"])
        self.usage = torch.zeros(shape, dtype=torch.int64)
        self.usage_runs = 0
        try:
            with open(self.usage_path) as f:
                saved = json.load(f)
        except FileNotFoundError:
            return
        counts = torch.tensor(saved["counts"], dtype=torch.int64)
        if counts.shape != shape:
            print(f"note: {self.usage_path} is for a different model shape; starting a new usage table")
            return
        self.usage, self.usage_runs = counts, saved.get("runs", 0)

    def choose_hot(self):
        """Picks the hot set: in each slab group, its HOT_SHARE of slots (leaving at least GPU_SLAB_MIN for other
        reads) go to the experts its layers have been routed to most, by the usage table."""
        self.hot = set()
        if not self.usage_runs:
            return
        for group in {id(g): g for g in self.slabs.values()}.values():
            slots = len(group["free"]) + len(group["lru"]) + len(group["hot_lru"])
            room = min(int(slots * HOT_SHARE), slots - GPU_SLAB_MIN)
            if room <= 0:
                continue
            layers = group["layers"]
            counts = self.usage[layers].flatten()
            top = counts.topk(min(room, counts.numel()))
            num_experts = self.usage.shape[1]
            self.hot.update((layers[i // num_experts], i % num_experts)
                            for i, used in zip(top.indices.tolist(), top.values.tolist()) if used)

    def record_usage(self, layer: int, selected: torch.Tensor):
        """Called by the engine with the experts each token was routed to."""
        self.usage[layer] += torch.bincount(selected.flatten(), minlength=self.usage.shape[1])

    def save_usage(self):
        """Writes the table (to a temp file, then renamed over the old one, so an interrupted write can't lose it)."""
        self.usage_runs += 1
        tmp = self.usage_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"runs": self.usage_runs, "counts": self.usage.tolist()}, f)
        os.replace(tmp, self.usage_path)

    def most_used(self, layer: int, n: int = None) -> list:
        """The layer's experts, most used first (the top n, if given), as (expert, count) pairs."""
        counts, order = self.usage[layer].sort(descending=True, stable=True)
        return list(zip(order.tolist(), counts.tolist()))[:n]

    def preload(self, fraction: float = PRELOAD_FRACTION):
        """Reads the most used experts (by the usage table) into the cache before generating, so they start as
        hits. With slabs, that's the hot set (see choose_hot). Otherwise (the plain LRU), each layer's most used, at
        most fraction of its experts, only ones that have been used, and no more than the layer's share of the
        cache; each layer's least used goes in first, so it is evicted first."""
        if not self.usage_runs or not fraction:
            return
        if not self.requested_cache:
            print("note: expert cache is off (--cache-gb 0), so the most used experts aren't preloaded")
            return
        start = time.perf_counter()
        reads = deque()
        if self.slabs:
            for key in sorted(self.hot):  # in pack order, so the SSD reads mostly forward
                reads.append((key, *self.start_read(*key)))
        else:  # the plain LRU: fit every layer's share, by its largest expert
            layer_bytes = sum(max(self.manifest["experts"][f"{layer}_{e}"]["read_length"]
                                  for e in range(self.usage.shape[1])) for layer in range(self.usage.shape[0]))
            count = min(int(self.usage.shape[1] * fraction), self.cache_bytes // layer_bytes)
            for layer in range(self.usage.shape[0]):
                for expert, used in reversed(self.most_used(layer, count)):
                    if used:
                        reads.append(((layer, expert), *self.start_read(layer, expert)))
        loaded = len(reads)
        while reads:  # popped as they are used, so read blocks aren't kept alive by their futures
            key, future, token = reads.popleft()
            weights = self.finish_read(key, future.result(), token)
            for w in weights:
                if isinstance(w, Int4Weight):
                    w.to_gpu()
        release_staged()
        where = "as the hot set" if self.slabs else f"up to {count} per layer"
        print(f"preloaded {loaded} most used experts ({self.cached_bytes / 1024**3:.2f} GiB, {where}) in "
              f"{time.perf_counter() - start:.1f}s")

    def close(self):
        self.pool.shutdown()
        os.close(self.fd)
