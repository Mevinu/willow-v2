"""A small local chat page for the model (`main.py web`), using only the standard library.

The model runs on the main thread (its GPU code expects to be driven from one thread). The web server runs on a
background thread and hands the model jobs through a queue, so one reply is generated at a time. The conversation
lives in the engine's caches, as with `main.py chat`; the server also keeps its text, so a reloaded page shows it."""
import json
import os
import queue
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
# The files the page is made of: URL path -> (file in web/, content type)
FILES = {"/": ("index.html", "text/html; charset=utf-8"),
         "/vendor/marked.umd.js": ("vendor/marked.umd.js", "text/javascript; charset=utf-8")}

class Job:
    """One request for the model thread: a chat message, or a reset."""

    def __init__(self, kind: str, message: str = None):
        self.kind, self.message = kind, message
        self.out = queue.Queue()  # dicts for the page, then None when the job is done
        self.stop = threading.Event()

class Chat:
    """What the web threads and the model thread share."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        self.history = []    # [{"role": "user" | "assistant", "text": str}]
        self.current = None  # the job the model is working on

    def snapshot(self) -> dict:
        with self.lock:
            return {"model": self.model_name, "busy": self.current is not None,
                    "history": [dict(turn) for turn in self.history]}

def reply(engine, chat: Chat, job: Job, max_new_tokens: int):
    """Runs one chat message on the model thread, streaming the reply's text into job.out."""
    with chat.lock:
        chat.history.append({"role": "user", "text": job.message})
        turn = {"role": "assistant", "text": ""}
        chat.history.append(turn)

    def on_text(piece: str) -> bool:
        with chat.lock:
            turn["text"] += piece
        job.out.put({"text": piece})
        return job.stop.is_set()

    try:
        engine.continue_with(engine.prompt_ids(job.message), max_new_tokens, on_text=on_text)
    except Exception as e:
        # A failure partway through a forward pass leaves the caches half updated, so the conversation starts over
        traceback.print_exc()
        engine.reset()
        with chat.lock:
            chat.history.clear()
        job.out.put({"error": f"{type(e).__name__}: {e}. The conversation was reset."})
        return
    read, decode = engine.stats["prefill"], engine.stats["decode"]
    job.out.put({"done": True, "stopped": job.stop.is_set(), "tokens": decode["tokens"],
                 "tokens_per_second": decode["tokens"] / decode["seconds"] if decode["tokens"] else 0,
                 "read_seconds": read["seconds"], "context_tokens": engine.position})

def make_handler(chat: Chat):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # keep the terminal for the model's own messages

        def send_json(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path in FILES:
                name, content_type = FILES[self.path]
                with open(os.path.join(HERE, name), "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/state":
                self.send_json(200, chat.snapshot())
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/api/chat":
                message = str(self.read_json().get("message", "")).strip()
                if not message:
                    return self.send_json(400, {"error": "empty message"})
                job = Job("chat", message)
                chat.jobs.put(job)
                # The reply streams as one JSON object per line
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while (item := job.out.get()) is not None:
                    try:
                        self.wfile.write(json.dumps(item).encode() + b"\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        job.stop.set()  # the page went away: end the reply at the next token
                        break
            elif self.path == "/api/stop":
                with chat.lock:
                    if chat.current is not None:
                        chat.current.stop.set()
                self.send_json(200, {"ok": True})
            elif self.path == "/api/reset":
                job = Job("reset")
                chat.jobs.put(job)
                while job.out.get() is not None:
                    pass
                self.send_json(200, chat.snapshot())
            else:
                self.send_json(404, {"error": "not found"})

    return Handler

def run_web(model: str, max_new_tokens: int, pack_dir: str, think: bool, cache_gb: float, load,
            port: int = 8000, open_browser: bool = True):
    """Loads the model, serves the chat page on http://127.0.0.1:port, and runs the model's jobs until Ctrl-C."""
    from safetensor.engine import Qwen35MoEEngine
    engine = load(model, pack_dir, cache_gb)
    if not isinstance(engine, Qwen35MoEEngine):
        raise SystemExit("web needs a Qwen3.5/3.6 model (it uses their chat format)")
    engine.think = think
    chat = Chat(os.path.basename(model))
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(chat))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    print(f"\nchat page: {url}  (Ctrl-C here quits)")
    if open_browser:
        webbrowser.open(url)
    try:
        while True:
            job = chat.jobs.get()
            with chat.lock:
                chat.current = job
            if job.kind == "reset":
                engine.reset()
                with chat.lock:
                    chat.history.clear()
            else:
                reply(engine, chat, job, max_new_tokens)
            with chat.lock:
                chat.current = None
            job.out.put(None)
    except KeyboardInterrupt:  # mid-reply, this leaves the caches half updated, so it quits rather than continues
        print()
    finally:
        server.shutdown()
        engine.experts.save_usage()
