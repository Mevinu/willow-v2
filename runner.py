"""The terminal front ends of `main.py run` and `main.py chat`. They work with any engine that has the methods
MoEEngine has (generate, continue_with, prompt_ids, reset, stats, experts); `chat` also needs engine.chat."""
import sys
import time

class ReplyPrinter:
    """Prints a reply to the terminal as it streams (an on_text callback). With thinking on, the reply starts inside
    the <think> block the prompt opened: that part is printed dimmed, and the </think> tag is left out."""

    DIM, PLAIN = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")

    def __init__(self, thinking: bool):
        self.thinking = thinking
        self.answer_started = False
        if thinking:
            print(self.DIM, end="")

    def __call__(self, piece: str) -> bool:
        if self.thinking:
            thought, tag, piece = piece.partition("</think>")  # </think> is one token, so it arrives whole
            print(thought, end="", flush=True)
            if not tag:
                return False
            self.thinking = False
            print(self.PLAIN + "\n", end="")
        if not self.answer_started:  # the model puts blank lines after </think>
            piece = piece.lstrip("\n")
            self.answer_started = bool(piece)
        print(piece, end="", flush=True)
        return False

    def finish(self):
        """Ends the reply's line, and the dimming if it stopped mid-thought."""
        print(self.PLAIN if self.thinking else "")
        self.thinking = False

def run_chat(load, max_new_tokens: int, think: bool):
    """An interactive chat (`main.py chat`). The model keeps the whole conversation in its caches, so each reply
    only processes the new message. /reset starts a new conversation; /exit, Ctrl-D or Ctrl-C quits.
    load() builds the engine."""
    engine = load()
    if not engine.chat:
        raise SystemExit("chat needs a chat model (this one has no chat format)")
    engine.think = think
    print("\nchat: type a message and press Enter. /think turns thinking on or off, /reset starts over, "
          "/exit (or Ctrl-D) quits.")
    print(f"(thinking {'on' if think else 'off'})")
    printer = None
    try:
        while True:
            try:
                message = input("\n> ").strip()
            except EOFError:
                break
            if message == "/exit":
                break
            if message == "/reset":
                engine.reset()
                print("(new conversation)")
                continue
            if message == "/think":
                engine.think = not engine.think
                print(f"(thinking {'on' if engine.think else 'off'})")
                continue
            if not message:
                continue
            print()
            printer = ReplyPrinter(engine.think)
            engine.continue_with(engine.prompt_ids(message), max_new_tokens, on_text=printer)
            printer.finish()
            read, reply = engine.stats["prefill"], engine.stats["decode"]
            speed = f" at {reply['tokens'] / reply['seconds']:.1f} tok/s" if reply["tokens"] else ""
            print(f"\n[{reply['tokens']} tokens{speed}; read your message in {read['seconds']:.1f}s; "
                  f"conversation so far: {engine.position} tokens]")
    except KeyboardInterrupt:  # mid-reply, this leaves the caches half-updated, so it quits rather than continues
        if printer:
            printer.finish()
        print()
    finally:
        engine.experts.save_usage()

def run_prompt(load, prompt: str, max_new_tokens: int, think: bool):
    """Generates a reply to prompt, streaming it to stdout, then prints speed and I/O stats (`main.py run`).
    load() builds the engine."""
    start = time.perf_counter()
    engine = load()
    load_seconds = time.perf_counter() - start
    engine.think = think
    if engine.chat:
        print(f"> {prompt}\n")
    else:
        print(prompt, end="", flush=True)
    printer = ReplyPrinter(think and engine.chat)
    try:
        engine.generate(prompt, max_new_tokens, on_text=printer)
    finally:
        printer.finish()
    engine.experts.save_usage()
    print_stats(engine, load_seconds)

def print_stats(engine, load_seconds: float):
    print(f"\n--- stats ---\nload:     {load_seconds:.1f}s")
    for name, s in engine.stats.items():
        if s["tokens"] == 0:
            continue
        gib = s["bytes_read"] / 1024**3
        io_share = s["read_seconds"] / s["seconds"] * 100
        io_speed = gib / s["read_seconds"] if s["read_seconds"] else 0
        lookups = s["cache_hits"] + s["disk_reads"]
        hit_rate = s["cache_hits"] / lookups * 100 if lookups else 0
        print(f"{name + ':':9} {s['tokens']:4d} tokens in {s['seconds']:6.1f}s = {s['tokens'] / s['seconds']:6.2f} tok/s | "
              f"cache hits {hit_rate:3.0f}% | {gib:6.2f} GiB read ({gib / s['tokens']:.2f} GiB/token) | "
              f"disk {io_share:3.0f}% of time at {io_speed:.1f} GiB/s")
        if s["prefetches"]:
            print(f"{'':9} prefetch: {s['prefetches']} next-layer reads started, "
                  f"{s['prefetch_used'] / s['prefetches'] * 100:.0f}% of them used")
    cache = engine.experts
    print(f"cache:    {cache.cached_bytes / 1024**3:.2f} / {cache.cache_bytes / 1024**3:.2f} GiB used, "
          f"{len(cache.cache)} experts held")
    if engine.experts_routed:
        print(f"skipped:  {engine.experts_skipped} of {engine.experts_routed} routed experts while decoding "
              f"({engine.experts_skipped / engine.experts_routed * 100:.0f}%), weight below {engine.min_expert_weight}")

    decode = engine.stats.get("decode")
    if decode and decode["tokens"]:
        print("\ndecode time per token, by section:")
        for section, seconds in sorted(decode["timings"].items(), key=lambda kv: -kv[1]):
            ms = seconds / decode["tokens"] * 1000
            print(f"  {section:18s} {ms:8.1f} ms  ({seconds / decode['seconds'] * 100:4.1f}%)")
