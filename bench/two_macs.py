"""Measures what running experts on a second Mac depends on, before building it:

  disk     How fast this Mac's SSD serves expert-sized random reads (8 at a time, OS file cache off, as
           ExpertStore reads), from a pack or from a test file it writes.
  server   Answers `net` from the other Mac (run it on the second Mac).
  net      Round-trip time and throughput to `server`: per layer, the main Mac would send a token's hidden state
           and expert ids (~10 KB) and get one result back (~5 KB), 48 times per token.
  metal    Whether willow's Metal kernels compile and give the same weights as the CPU dequantizers on this Mac
           (needs willow's venv: torch).

disk, server and net use Python's standard library only, so they run with the system python3 before willow is
installed. Run from the willow-v02 folder, e.g.:
  python3 bench/two_macs.py disk --make 8           (writes an 8 GiB test file, then reads it)
  python3 bench/two_macs.py server                  (on the second Mac)
  python3 bench/two_macs.py net 169.254.12.34       (the second Mac's Thunderbolt Bridge address)
  .venv/bin/python bench/two_macs.py metal"""
import argparse
import fcntl
import os
import random
import socket
import statistics
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

EXPERT_BYTES = 2435072  # Qwen3.8 UD-IQ4_XS: an expert's average read (gate_up + down), 16 KiB aligned
ALIGN = 16384
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
TEST_FILE = os.path.expanduser("~/willow-disk-test.bin")
PORT = 8765

# disk

def make_file(path: str, gib: float):
    """Writes gib GiB of random bytes (random so nothing can compress or dedupe them), past the OS file cache."""
    chunk = os.urandom(64 << 20)
    total = int(gib * (1 << 30))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    start, written = time.perf_counter(), 0
    try:
        while written < total:
            # Change the chunk a little each time, so no two are the same
            os.write(fd, struct.pack("<Q", written) + chunk[8:min(len(chunk), total - written)])
            written += min(len(chunk), total - written)
            print(f"\rwriting {path}: {written / (1 << 30):.1f} / {gib:g} GiB", end="", flush=True)
    finally:
        os.close(fd)
    print(f"  ({written / (1 << 30) / (time.perf_counter() - start):.1f} GiB/s)")

def disk(args):
    path = args.path or TEST_FILE
    if args.make:
        make_file(path, args.make)
    if not os.path.exists(path):
        sys.exit(f"{path} doesn't exist: pass a pack's experts.pack, or --make GIB to write a test file")
    size = os.path.getsize(path)
    if size < 4 * EXPERT_BYTES:
        sys.exit(f"{path} is too small")
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)  # as ExpertStore does: every read comes from the SSD
    slots = (size - args.size) // ALIGN
    lock, count = threading.Lock(), [0]
    deadline = time.perf_counter() + args.seconds

    def worker():
        buf = bytearray(args.size)
        rng = random.Random()
        while time.perf_counter() < deadline:
            got = os.preadv(fd, [buf], rng.randrange(slots) * ALIGN)
            if got != args.size:
                raise IOError(f"short read: {got}")
            with lock:
                count[0] += 1

    start = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as pool:
        for future in [pool.submit(worker) for _ in range(args.threads)]:
            future.result()
    seconds = time.perf_counter() - start
    os.close(fd)
    gib = count[0] * args.size / (1 << 30)
    print(f"disk: {count[0]} random reads of {args.size / (1 << 20):.2f} MiB, {args.threads} at a time, "
          f"in {seconds:.1f}s = {gib / seconds:.2f} GiB/s ({count[0] / seconds:.0f} experts/s)")
    if args.make and not args.keep:
        os.remove(path)
        print(f"removed {path}")

# net: each message is a 4-byte length, then that many bytes; the server answers each with reply_bytes

def recv_exact(sock, n: int) -> bytes:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("connection closed")
        got += k
    return bytes(buf)

def serve_one(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with conn:
        try:
            while True:
                length, reply = struct.unpack("<II", recv_exact(conn, 8))
                recv_exact(conn, length)
                conn.sendall(b"\0" * reply)
        except ConnectionError:
            pass

def server(args):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.port))
    listener.listen()
    print(f"listening on {args.bind}:{args.port} (Ctrl-C stops). This Mac's addresses:")
    # A Thunderbolt link gets a self-assigned 169.254.x.x address: on bridge0 (the Thunderbolt Bridge service),
    # or on the port's own enN when each port is its own service (as exo sets it up)
    for interface in os.popen("ifconfig -l").read().split():
        address = os.popen(f"ipconfig getifaddr {interface} 2>/dev/null").read().strip()
        if address:
            cable = address.startswith("169.254.")
            print(f"  {interface}: {address}" + ("  <- the cable (use this one)" if cable else ""))
    try:
        while True:
            conn, peer = listener.accept()
            print(f"connection from {peer[0]}")
            threading.Thread(target=serve_one, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass

def net(args):
    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def round_trip(send: int, reply: int) -> float:
        start = time.perf_counter()
        sock.sendall(struct.pack("<II", send, reply) + b"\0" * send)
        recv_exact(sock, reply)
        return time.perf_counter() - start

    for _ in range(50):  # warm up
        round_trip(10240, 5120)
    times = sorted(round_trip(10240, 5120) for _ in range(args.count))
    median, p99 = statistics.median(times), times[int(len(times) * 0.99)]
    print(f"net: {args.count} round trips of 10 KB out, 5 KB back: median {median * 1e3:.3f} ms, "
          f"99th percentile {p99 * 1e3:.3f} ms")
    print(f"     48 per token (one per layer): {48 * median * 1e3:.1f} ms per token")
    big, sent, start = 16 << 20, 0, time.perf_counter()
    while sent < (1 << 30):
        round_trip(big, 8)
        sent += big
    seconds = time.perf_counter() - start
    print(f"     throughput: 1 GiB in {seconds:.2f}s = {sent / seconds / (1 << 30):.2f} GiB/s "
          f"({sent * 8 / seconds / 1e9:.1f} Gbit/s)")
    sock.close()

# metal

def metal(args):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from gguf import metal as kernels, quants
    from gguf.reader import BLOCK_SIZES
    import platform
    print(f"macOS {platform.mac_ver()[0]}, torch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
    start = time.perf_counter()
    kernels.library()
    print(f"metal: compiled the kernels in {time.perf_counter() - start:.1f}s")
    rows, k = 64, 512
    ok = True
    for type_name in kernels.TYPES:
        elements, block_bytes = BLOCK_SIZES[type_name]
        raw = torch.randint(0, 256, (rows * k // elements * block_bytes,), dtype=torch.uint8)
        cpu = quants.dequantize(raw, type_name, (rows, k))
        gpu = kernels.dequantize_gpu(raw.to("mps"), type_name, (rows, k)).cpu()
        same = bool(((cpu == gpu) | (cpu.isnan() & gpu.isnan())).all())  # random bytes make some NaN scales
        x = torch.randn(1, k, dtype=torch.bfloat16)
        y = kernels.gemv(raw.to("mps"), type_name, (rows, k), x.to("mps")).cpu()
        ok &= same and y.shape == (1, rows)
        print(f"  {type_name:8} dequantize {'matches the CPU' if same else 'DIFFERS from the CPU'}, gemv runs")
    print("metal: all kernels work" if ok else "metal: some kernels differ")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("disk", help="random expert-sized reads from the SSD")
    d.add_argument("path", nargs="?", help=f"file to read (a pack's experts.pack; default {TEST_FILE})")
    d.add_argument("--make", type=float, metavar="GIB", help="first write a test file of this many GiB")
    d.add_argument("--keep", action="store_true", help="keep the test file afterwards")
    d.add_argument("--size", type=int, default=EXPERT_BYTES, help="bytes per read")
    d.add_argument("--threads", type=int, default=8)
    d.add_argument("--seconds", type=float, default=10)
    s = commands.add_parser("server", help="answer `net` (run on the second Mac)")
    s.add_argument("--bind", default="0.0.0.0")
    s.add_argument("--port", type=int, default=PORT)
    n = commands.add_parser("net", help="round trips and throughput to `server`")
    n.add_argument("host")
    n.add_argument("--port", type=int, default=PORT)
    n.add_argument("--count", type=int, default=2000)
    commands.add_parser("metal", help="check willow's Metal kernels on this Mac")
    args = parser.parse_args()
    {"disk": disk, "server": server, "net": net, "metal": metal}[args.command](args)

if __name__ == "__main__":
    main()
