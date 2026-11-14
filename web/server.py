"""A small local chat page for the model (`main.py web`), using only the standard library.

The model runs on the main thread (its GPU code expects to be driven from one thread). The web server runs on a
background thread and hands the model jobs through a queue, so one reply is generated at a time. The conversation
lives in the engine's caches, as with `main.py chat`; the server also keeps its text, so a reloaded page shows it.

The page can switch between the models in models.json. A switch is a job like the others: the model thread frees
the current model (its memory has to be gone before the next one fits), loads the new one and starts a new
conversation. If the new one fails to load, it goes back to the previous one."""
import gc
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
    """One request for the model thread: a chat message (message: its text), a reset, or a switch (message: the
    model's name)."""

    def __init__(self, kind: str, message: str = None, think: bool = None):
        self.kind, self.message = kind, message
        self.think = think  # think before answering; None keeps the engine's setting
        self.out = queue.Queue()  # dicts for the page, then None when the job is done
        self.stop = threading.Event()

class Chat:
    """What the web threads and the model thread share."""

    def __init__(self, models: list, model_name: str, think: bool):
        self.models = {entry["name"]: entry for entry in models}  # name -> models.json entry, in its order
        self.model_name = model_name
        self.loading = None  # the model being switched to
        self.think = think   # thinking for the next message (the last one's setting, so a reloaded page keeps it)
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        self.history = []    # [{"role": "user" | "assistant", "text": str}]
        self.current = None  # the job the model is working on

    def snapshot(self) -> dict:
        with self.lock:
            return {"model": self.model_name, "models": list(self.models), "loading": self.loading,
                    "busy": self.current is not None, "think": self.think,
                    "history": [dict(turn) for turn in self.history]}

def reply(engine, chat: Chat, job: Job, max_new_tokens: int):
    """Runs one chat message on the model thread, streaming the reply's text into job.out. With thinking on, the
    prompt ends with <think>, so the reply starts mid-thought: its text gets the tag back, for the page to fold."""
    if job.think is not None:
        engine.think = job.think
    ids = engine.prompt_ids(job.message)
    with chat.lock:
        chat.think = engine.think
        chat.history.append({"role": "user", "text": job.message})
        turn = {"role": "assistant", "text": "<think>" if engine.think else ""}
        chat.history.append(turn)
    if turn["text"]:
        job.out.put({"text": turn["text"]})

    def on_text(piece: str) -> bool:
        with chat.lock:
            turn["text"] += piece
        job.out.put({"text": piece})
        return job.stop.is_set()

    try:
        engine.continue_with(ids, max_new_tokens, on_text=on_text)
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

class ModelHost:
    """The loaded engine, on the model thread. load(entry) builds an engine from a models.json entry."""

    def __init__(self, load, entry: dict):
        self.load_entry = load
        self.engine = None
        self.load(entry)

    def load(self, entry: dict):
        engine = self.load_entry(entry)
        if not engine.chat:
            engine.experts.close()
            raise ValueError(f"{entry['name']} has no chat format, so it can't be used in the chat page")
        self.engine = engine

    def unload(self):
        """Saves the expert usage table and frees the engine, its expert cache and its GPU memory."""
        if self.engine is None:
            return
        self.engine.experts.save_usage()
        self.engine.experts.close()
        self.engine = None
        gc.collect()
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def switch(host: ModelHost, chat: Chat, job: Job):
    """Runs a switch job on the model thread."""
    name = job.message
    if name == chat.model_name:
        job.out.put({"done": True})
        return
    previous = chat.models[chat.model_name]
    with chat.lock:
        chat.loading = name
        chat.history.clear()
    print(f"\nswitching to {name}")
    host.unload()
    error = None
    try:
        host.load(chat.models[name])
    except (Exception, SystemExit) as e:  # check_gguf exits with SystemExit on a missing file
        traceback.print_exc()
        error = f"Couldn't load {name}: {e}. Back to {previous['name']}."
        print(f"\n{error}")
        host.load(previous)  # if this fails too, the server stops
        name = previous["name"]
    host.engine.think = chat.think
    with chat.lock:
        chat.model_name, chat.loading = name, None
    job.out.put({"error": error} if error else {"done": True})

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
                body = self.read_json()
                message = str(body.get("message", "")).strip()
                if not message:
                    return self.send_json(400, {"error": "empty message"})
                think = body.get("think")
                job = Job("chat", message, None if think is None else bool(think))
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
            elif self.path == "/api/model":
                name = self.read_json().get("name")
                if name not in chat.models:
                    return self.send_json(400, {"error": f"unknown model {name!r}"})
                job = Job("switch", name)
                chat.jobs.put(job)
                result = {}
                while (item := job.out.get()) is not None:
                    result.update(item)
                self.send_json(200, dict(chat.snapshot(), error=result.get("error")))
            elif self.path == "/api/reset":
                job = Job("reset")
                chat.jobs.put(job)
                while job.out.get() is not None:
                    pass
                self.send_json(200, chat.snapshot())
            else:
                self.send_json(404, {"error": "not found"})

    return Handler

def run_web(models: list, model_name: str, max_new_tokens: int, think: bool, load, port: int = 8000,
            open_browser: bool = True):
    """Loads the model named model_name (one of models, models.json's entries), serves the chat page on
    http://127.0.0.1:port, and runs the model's jobs until Ctrl-C. load(entry) builds an engine from an entry."""
    chat = Chat(models, model_name, think)
    try:
        host = ModelHost(load, chat.models[model_name])
    except ValueError as e:
        raise SystemExit(str(e))
    host.engine.think = think
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
                host.engine.reset()
                with chat.lock:
                    chat.history.clear()
            elif job.kind == "switch":
                switch(host, chat, job)
            else:
                reply(host.engine, chat, job, max_new_tokens)
            with chat.lock:
                chat.current = None
            job.out.put(None)
    except KeyboardInterrupt:  # mid-reply, this leaves the caches half updated, so it quits rather than continues
        print()
    finally:
        server.shutdown()
        if host.engine is not None:
            host.engine.experts.save_usage()
