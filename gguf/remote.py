"""Runs part of each layer's experts on a second Mac while decoding, so both Macs' SSDs read at once.

Decoding is limited by how fast experts are read from the SSD (~1.2 GiB per token for Qwen3.8). A second Mac
connected by a Thunderbolt/USB4 cable (a round trip there takes ~0.1 ms) adds its own SSD:
- `main.py split-pack` picks, for each layer, the experts the second Mac will run (by the usage table, so it gets
  about `share` of the layer's reads) and writes them into a smaller pack, to be copied to it.
- `main.py serve-experts` (on the second Mac) loads that pack, with its own expert cache, and runs its experts with
  the same Metal kernels (metal.decode_experts).
- `--remote HOST` (on this Mac): at each layer of a decode step, the engine sends the second Mac the token's hidden
  state, which of its experts the token was routed to (with their routing weights), and its share of the guesses
  for the next layer; then reads and runs its own experts while the second Mac does the same, and adds the second
  Mac's result (their routing-weighted sum) to its own. ~10 KB out and 5 KB back per layer.
This Mac keeps the whole pack: a prompt's prefill, layers the kernels don't run, and everything after the second Mac
goes away run here as before. The result can differ from one Mac's in the last bf16 bit (the two halves are summed
separately, then added), so greedy decoding can drift from a one-Mac run after many tokens.

The protocol is length-prefixed binary messages over one TCP connection (TCP_NODELAY):
  request  HELLO:  B type=0                                  -> I length, JSON {owned, parts, num_layers, ...}
  request  LAYER:  B type=1, H layer, H n, H g, I x_bytes; then n x H experts, n x 2 bytes of bf16 routing
                   weights, g x H guessed experts for layer + 1, x_bytes of bf16 hidden state
                   -> (only when n > 0) B status (0: ok), I length, then the bf16 [hidden] result, or error text
  request  BYE:    B type=2                                  (the server saves its usage table)"""
import atexit
import json
import os
import socket
import struct
import time

import torch

from .compiler import PAGE_ALIGNMENT, align_up

PORT = 8765
HELLO, LAYER, BYE = 0, 1, 2
LAYER_HEADER = struct.Struct("<BHHHI")
REPLY_HEADER = struct.Struct("<BI")
DEFAULT_SHARE = 0.33  # the M1 13" Pro's SSD (3.2 GiB/s) next to an M5's (6.5): about a third of the reads

def recv_exact(sock, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("the other Mac closed the connection")
        got += k
    return buf

def tensor_bytes(t: torch.Tensor) -> bytes:
    """A CPU tensor's raw bytes (without NumPy, which the second Mac may not have)."""
    buf = bytearray(t.numel() * t.element_size())
    torch.frombuffer(buf, dtype=torch.uint8).copy_(t.contiguous().view(-1).view(torch.uint8))
    return bytes(buf)

def parse_address(address: str) -> tuple:
    host, _, port = address.rpartition(":") if ":" in address else (address, "", "")
    return host, int(port) if port else PORT

# Splitting a pack

def choose_owned(usage: torch.Tensor, share: float) -> list:
    """For each layer, the experts the second Mac runs: in order of use (most first), each goes wherever keeps
    the second Mac's part of the layer's uses closest to share, so hot and cold experts are split alike.
    Every expert counts at least 1 (unused experts are split by number). -> [sorted experts] per layer."""
    owned = []
    for counts in usage.tolist():
        order = sorted(range(len(counts)), key=lambda e: (-counts[e], e))
        remote_uses = total = 0
        mine = []
        for expert in order:
            uses = counts[expert] + 1
            total += uses
            if remote_uses + uses <= share * total + uses / 2:
                mine.append(expert)
                remote_uses += uses
        owned.append(sorted(mine))
    return owned

def split_pack(pack_dir: str, out_dir: str, share: float = DEFAULT_SHARE):
    """Writes the second Mac's pack to out_dir: its experts' blocks copied as they are (16 KiB aligned), a manifest
    listing only them, and the usage table (so it can preload its most used ones)."""
    from .store import USAGE_FILE
    with open(os.path.join(pack_dir, "manifest.json")) as f:
        manifest = json.load(f)
    shape = (manifest["num_layers"], manifest["num_experts"])
    try:
        with open(os.path.join(pack_dir, USAGE_FILE)) as f:
            saved = json.load(f)
        usage = torch.tensor(saved["counts"], dtype=torch.int64)
    except FileNotFoundError:
        saved, usage = {"runs": 0}, torch.zeros(shape, dtype=torch.int64)
        print("note: no usage table yet (run the model once first), so experts are split by number only")
    owned = choose_owned(usage, share)
    os.makedirs(out_dir, exist_ok=True)
    experts, offset = {}, 0
    src = os.open(os.path.join(pack_dir, "experts.pack"), os.O_RDONLY)
    total = sum(manifest["experts"][f"{layer}_{e}"]["read_length"]
                for layer, mine in enumerate(owned) for e in mine)
    start, done = time.perf_counter(), 0
    with open(os.path.join(out_dir, "experts.pack"), "wb") as out:
        for layer, mine in enumerate(owned):
            for expert in mine:
                entry = dict(manifest["experts"][f"{layer}_{expert}"])
                block = os.pread(src, entry["read_length"], entry["file_offset"])
                out.seek(offset)
                out.write(block)
                entry["file_offset"] = offset
                experts[f"{layer}_{expert}"] = entry
                offset = align_up(offset + len(block), PAGE_ALIGNMENT)
                done += len(block)
            print(f"\rwriting {out_dir}: {done / 1024**3:.1f} / {total / 1024**3:.1f} GiB", end="", flush=True)
        out.truncate(offset)
    os.close(src)
    split = dict(manifest, experts=experts, split={"share": share, "source": os.path.abspath(pack_dir)})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(split, f)
    mask = torch.zeros(shape, dtype=torch.bool)
    for layer, mine in enumerate(owned):
        mask[layer, mine] = True
    with open(os.path.join(out_dir, USAGE_FILE), "w") as f:
        json.dump({"runs": saved.get("runs", 0), "counts": (usage * mask).tolist()}, f)
    uses = sum(int(usage[layer, mine].sum()) for layer, mine in enumerate(owned))
    print(f"\n{len(experts)} of {shape[0] * shape[1]} experts ({len(experts) / (shape[0] * shape[1]) * 100:.0f}%), "
          f"{total / 1024**3:.1f} GiB, in {time.perf_counter() - start:.0f}s; "
          + (f"they had {uses / max(1, int(usage.sum())) * 100:.0f}% of the recorded uses" if usage.sum() else ""))

# The second Mac's side

class ExpertServer:
    """Runs a split pack's experts for one connected engine at a time (see the module docstring)."""

    def __init__(self, pack_dir: str, cache_gb: float, preload: bool = True):
        from . import metal
        from .store import ExpertStore
        if not metal.available():
            raise SystemExit("serve-experts needs the GPU (MPS), which torch can't use here")
        self.store = ExpertStore(pack_dir, int(cache_gb * 1024**3))
        if not self.store.gpu:
            raise SystemExit(f"{pack_dir} isn't a GPU pack (`pack --gpu`), so its experts can't run on the kernels")
        manifest = self.store.manifest
        self.owned = self.store.held
        not_batched = [layer for layer, mine in enumerate(self.owned) if mine and not self.store.runs_batched(layer)]
        if not_batched:
            raise SystemExit(f"layers {not_batched} hold expert types the batched kernels don't run")
        if preload:
            self.store.preload()
        self.info = {"num_layers": manifest["num_layers"], "num_experts": manifest["num_experts"],
                     "parts": manifest["parts"], "owned": self.owned}
        self.event = torch.mps.Event()

    def serve(self, bind: str, port: int):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind, port))
        listener.listen(1)
        print(f"serving {sum(map(len, self.owned))} experts on port {port}. This Mac's cable address:")
        for interface in os.popen("ifconfig -l").read().split():
            address = os.popen(f"ipconfig getifaddr {interface} 2>/dev/null").read().strip()
            if address.startswith("169.254."):
                print(f"  {address}  (run with --remote {address} on the other Mac)")
        try:
            while True:
                conn, peer = listener.accept()
                print(f"connected: {peer[0]}")
                try:
                    self.handle(conn)
                except (ConnectionError, OSError) as error:
                    print(f"disconnected: {error}")
                finally:
                    conn.close()
                    self.store.save_usage()
        except KeyboardInterrupt:
            print()
        finally:
            self.store.close()

    def handle(self, conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        tokens, start = 0, time.perf_counter()
        store = self.store
        store.hits = store.reads = store.bytes_read = store.prefetches = store.prefetch_used = 0
        store.read_seconds = 0.0
        self.timings = {"idle (waiting for the other Mac)": 0.0}
        self.clock = time.perf_counter()
        while True:
            kind = recv_exact(conn, 1)[0]
            self.tick("idle (waiting for the other Mac)")
            if kind == HELLO:
                body = json.dumps(self.info).encode()
                conn.sendall(struct.pack("<I", len(body)) + body)
            elif kind == BYE:
                self.report(tokens, time.perf_counter() - start)
                return
            elif kind == LAYER:
                _, layer, n, g, x_bytes = LAYER_HEADER.unpack(bytes([kind]) + recv_exact(conn, LAYER_HEADER.size - 1))
                body = recv_exact(conn, 2 * n + 2 * n + 2 * g + x_bytes)
                experts = list(struct.unpack_from(f"<{n}H", body))
                guesses = list(struct.unpack_from(f"<{g}H", body, 4 * n))
                tokens += layer == 0
                if not n:
                    self.prefetch(layer + 1, guesses)
                    continue
                try:
                    result = self.run(layer, experts, body[2 * n:4 * n], guesses, body[4 * n + 2 * g:])
                    conn.sendall(REPLY_HEADER.pack(0, len(result)) + result)
                    self.tick("send reply")
                except (ConnectionError, OSError):
                    raise
                except Exception as error:  # the engine then runs these experts itself
                    import traceback
                    traceback.print_exc()
                    text = f"{type(error).__name__}: {error}".encode()
                    conn.sendall(REPLY_HEADER.pack(1, len(text)) + text)
            else:
                raise ConnectionError(f"unknown message type {kind}")

    def tick(self, section: str):
        now = time.perf_counter()
        self.timings[section] = self.timings.get(section, 0.0) + now - self.clock
        self.clock = now

    def report(self, tokens: int, seconds: float):
        """Prints where the time went, per decode step, and the disk and cache figures."""
        print(f"done: {tokens} decode steps in {seconds:.0f}s")
        if not tokens:
            return
        for section, total in sorted(self.timings.items(), key=lambda kv: -kv[1]):
            print(f"  {section:34s} {total / tokens * 1000:7.1f} ms per step")
        store = self.store
        gib = store.bytes_read / 1024**3
        lookups = store.hits + store.reads
        print(f"  {gib / tokens:.2f} GiB read per step, disk busy {store.read_seconds / seconds * 100:.0f}% of the time "
              f"at {gib / store.read_seconds if store.read_seconds else 0:.1f} GiB/s; cache hits "
              f"{store.hits / lookups * 100 if lookups else 0:.0f}%; prefetch: {store.prefetches} reads, "
              f"{store.prefetch_used / store.prefetches * 100 if store.prefetches else 0:.0f}% used")

    def prefetch(self, layer: int, guesses: list):
        if guesses and layer < len(self.owned):
            self.store.prefetch(layer, guesses)

    def run(self, layer: int, experts: list, weights: bytes, guesses: list, x: bytes) -> bytes:
        from . import metal
        order = sorted(range(len(experts)), key=experts.__getitem__)
        experts = [experts[i] for i in order]
        self.tick("receive request")
        loads = self.store.load_batches(layer, experts)
        self.prefetch(layer + 1, guesses)  # behind this layer's reads
        gpu = torch.device("mps")
        x = torch.frombuffer(bytearray(x), dtype=torch.bfloat16).view(1, -1).to(gpu)
        weights = torch.frombuffer(bytearray(weights), dtype=torch.bfloat16).to(gpu)
        self.tick("read setup")
        out = metal.decode_experts(x, experts, loads, weights, order, metal.expert_gate, self.event, self.tick)
        self.store.record_usage(layer, torch.tensor(experts))
        out = out.cpu()
        self.tick("gpu wait")
        return tensor_bytes(out)

# This Mac's side

class RemoteExperts:
    """The engine's connection to a second Mac's expert server. owned[layer] is the set of experts it runs."""

    def __init__(self, address: str, manifest: dict):
        host, port = parse_address(address)
        self.address = f"{host}:{port}"
        # A short wait to connect (a Mac that's asleep or unplugged doesn't answer), then up to 10 s per message
        self.sock = socket.create_connection((host, port), timeout=3)
        self.sock.settimeout(10)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.sendall(bytes([HELLO]))
        info = json.loads(recv_exact(self.sock, struct.unpack("<I", recv_exact(self.sock, 4))[0]))
        if (info["num_layers"], info["num_experts"]) != (manifest["num_layers"], manifest["num_experts"]):
            raise ValueError(f"{self.address} serves a different model ({info['num_layers']} layers, "
                             f"{info['num_experts']} experts)")
        for layer, mine in enumerate(info["owned"]):
            for name, part in manifest["parts"].items():
                if mine and name.split(".")[0] == str(layer) and info["parts"].get(name) != part:
                    raise ValueError(f"{self.address}'s pack doesn't match this one (part {name}); split and copy "
                                     "it again")
        self.owned = [set(mine) for mine in info["owned"]]
        self.alive = True
        self.sent = self.experts_sent = 0  # replies owed; experts the other Mac has run
        self.hidden_bytes = None
        atexit.register(self.close)  # so the server prints its summary and saves its usage table

    def send(self, layer: int, experts: list, weights: list, guesses: list, x: torch.Tensor = None):
        """experts (with their routing weights, as raw bf16 bit patterns) for the other Mac to run on x (a CPU bf16
        [1, hidden], given when there are experts), and guesses for layer + 1 for it to start reading."""
        x_bytes = b"" if x is None else tensor_bytes(x)
        self.hidden_bytes = len(x_bytes) or self.hidden_bytes
        n, g = len(experts), len(guesses)
        self.sock.sendall(LAYER_HEADER.pack(LAYER, layer, n, g, len(x_bytes))
                          + struct.pack(f"<{n}H{n}H{g}H", *experts, *[w & 0xFFFF for w in weights], *guesses)
                          + x_bytes)
        if n:
            self.sent += 1
            self.experts_sent += n

    def receive(self) -> torch.Tensor:
        """The other Mac's answer to the last send with experts: their routing-weighted sum, bf16 [1, hidden] on
        the CPU."""
        self.sent -= 1
        status, length = REPLY_HEADER.unpack(recv_exact(self.sock, REPLY_HEADER.size))
        body = recv_exact(self.sock, length)
        if status:
            raise RuntimeError(f"the other Mac couldn't run its experts: {body.decode(errors='replace')}")
        return torch.frombuffer(body, dtype=torch.bfloat16).view(1, -1)

    def fail(self, error: Exception):
        """Stops using the other Mac (this one runs every expert from now on)."""
        if self.alive:
            print(f"\nnote: lost the expert server at {self.address} ({error}); running every expert here")
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass

    def close(self):
        if self.alive:
            try:
                self.sock.sendall(bytes([BYE]))
            except OSError:
                pass
            self.sock.close()
            self.alive = False
