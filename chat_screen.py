"""The full-window layout of `main.py chat`: the conversation scrolls in the top part of the terminal, and the bottom
rows stay where they are: a status line with the live token speed, and a box to type the next message in.

    ...the conversation, scrolling...
     12.4 tok/s  (avg 11.9) · 132 tokens · 11.1s · context 2,563             Ctrl-C quits
    ╭──────────────────────────────────────────────────────────────────────────────────╮
    │ › your next message, which can run over                                          │
    │   several lines                                                                  │
    ╰─────────────────────────────────────────────── Enter sends · Ctrl-J new line ─╯

Plain ANSI escape codes, so nothing extra needs installing: a scroll region keeps the conversation above the bottom
rows, and every other write goes there through absolute cursor moves. While it is open, sys.stdout is replaced by
a stream that writes into the conversation, so ReplyPrinter (and any note the engine prints) needs no changes.
The conversation's cursor is the one saved with DECSC (ESC 7): every write to it restores it first and saves it
after, and drawing the bottom rows ends by restoring it.

The box is a small editor of its own (Editor, Keys), in cbreak mode: it wraps long lines and grows upward with the
message (up to MAX_INPUT_ROWS, then scrolls), keeps the newlines of pasted text (bracketed paste), and Enter sends
while Ctrl-J, Alt+Enter or Shift+Enter (where the terminal tells it apart) starts a new line. While the box is
taller than one row, the conversation's scroll region is shorter; the box only pushes the conversation up when it
would cover its last line, which it finds by asking the terminal where the cursor is (DSR).

A resize is redrawn from scratch: terminals re-wrap what's on the screen when its width changes, which moves the
old status line and box into the conversation and leaves the saved cursor pointing at the wrong place. So the
screen keeps a transcript of everything written to the conversation, and once the window has stopped changing size
(RESIZE_SETTLE), it clears the screen and the scrollback and prints the transcript again at the new width. Until
then nothing is drawn (the reply's text still goes into the transcript).

Esc stops a reply after the token being written, like the web page's Stop: while a reply is written, the terminal
is in cbreak mode too and each token checks for a key press. Esc in the box clears it."""
import codecs
import os
import re
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
import unicodedata
from collections import deque

CSI = "\033["
SAVE, RESTORE = "\0337", "\0338"  # the conversation's cursor (position and text style)
RESET, BOLD, DIM, GREEN = CSI + "0m", CSI + "1m", CSI + "2m", CSI + "32m"
REFRESH_SECONDS = 0.25     # how often the status line is redrawn while the model works
RESIZE_SETTLE = 0.15       # a resize is handled once the window has kept its size this long (a drag sends many)
TRANSCRIPT_LIMIT = 4 << 20  # characters of the conversation kept for redrawing it after a resize
SPEED_WINDOW = 16          # the live speed is over this many most recent tokens
MAX_INPUT_ROWS = 10        # the box grows with the message up to this many rows, then scrolls
MIN_CONVERSATION_ROWS = 3  # rows the conversation keeps however tall the box gets
ESC_WAIT = 0.05            # after an ESC byte, how long the rest of a key's sequence may take before it counts as Esc
TAB = "    "               # a tab is typed (and pasted) as spaces, so every character's width is known
BOX_HINT = " Enter sends · Ctrl-J new line "
DSR_REPLY = re.compile(rb"\x1b\[(\d+);(\d+)R")  # the terminal's answer to "where is the cursor?" (CSI 6n)

def char_width(ch: str) -> int:
    """Columns a character takes in the terminal: 2 for wide (CJK, most emoji), 0 for combining marks."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1

def text_width(text: str) -> int:
    return sum(char_width(ch) for ch in text)

def wrap(text: str, width: int) -> list:
    """Splits text into the rows the box shows, width columns each: at every newline, and wherever a line runs
    out of room, after its last space if the row has one (a word longer than a row is cut). -> [(start, end)]
    indexes into text, newlines left out; a row that breaks at a space ends with it."""
    rows, start = [], 0
    for line_end in [i for i, ch in enumerate(text) if ch == "\n"] + [len(text)]:
        row_start, used, space = start, 0, None  # space: just after the row's last space
        for i in range(start, line_end):
            w = char_width(text[i])
            if used + w > width and i > row_start:
                if text[i] == " ":
                    pass  # the space that ends this row may hang past its edge; the next row starts after it
                elif space is not None and space > row_start and text_width(text[space:i]) + w <= width:
                    rows.append((row_start, space))
                    row_start, used = space, text_width(text[space:i])
                else:  # no space to break at, or the word after it doesn't fit a row either: cut here
                    rows.append((row_start, i))
                    row_start, used = i, 0
                if text[i] == " ":
                    rows.append((row_start, i + 1))
                    row_start, used, space = i + 1, 0, None
                    continue
            used += w
            if text[i] == " ":
                space = i + 1
        rows.append((row_start, line_end))
        start = line_end + 1
    return rows

def locate(rows: list, text: str, cursor: int) -> tuple:
    """-> (the row the cursor is on, its column there). Where a long line wraps, the cursor at the break shows at
    the start of the next row."""
    row = max(i for i, (start, end) in enumerate(rows) if start <= cursor <= end)
    return row, text_width(text[rows[row][0]:cursor])

class Editor:
    """The message being typed in the box: its text, the cursor (an index into it), and the earlier messages
    (up and down on the first or last row go through them, like a shell's history)."""

    def __init__(self):
        self.history = []
        self.clear()

    def clear(self):
        self.text, self.cursor = "", 0
        self.history_pos, self.draft = len(self.history), ""

    def submit(self) -> str:
        text = self.text
        if text.strip() and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self.clear()
        return text

    def insert(self, s: str):
        self.text = self.text[:self.cursor] + s + self.text[self.cursor:]
        self.cursor += len(s)

    def backspace(self):
        if self.cursor:
            self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
            self.cursor -= 1

    def delete(self):
        self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]

    def move(self, delta: int):
        self.cursor = max(0, min(len(self.text), self.cursor + delta))

    def line_start(self) -> int:
        return self.text.rfind("\n", 0, self.cursor) + 1

    def line_end(self) -> int:
        end = self.text.find("\n", self.cursor)
        return len(self.text) if end < 0 else end

    def word_left(self) -> int:
        i = self.cursor
        while i and not self.text[i - 1].isalnum():
            i -= 1
        while i and self.text[i - 1].isalnum():
            i -= 1
        return i

    def word_right(self) -> int:
        i, n = self.cursor, len(self.text)
        while i < n and not self.text[i].isalnum():
            i += 1
        while i < n and self.text[i].isalnum():
            i += 1
        return i

    def cut(self, start: int, end: int):
        """Deletes text[start:end], leaving the cursor at start."""
        self.text = self.text[:start] + self.text[end:]
        self.cursor = start

    def vertical(self, width: int, direction: int) -> bool:
        """Moves the cursor to the row above (-1) or below (+1), at the same column as far as that row allows.
        -> False when there is no such row."""
        rows = wrap(self.text, width)
        row, column = locate(rows, self.text, self.cursor)
        target = row + direction
        if not 0 <= target < len(rows):
            return False
        start, end = rows[target]
        i, used = start, 0
        while i < end and used + char_width(self.text[i]) <= column:
            used += char_width(self.text[i])
            i += 1
        if i == end and end > start and target + 1 < len(rows) and rows[target + 1][0] == end:
            i -= 1  # the end of a wrapped row would show on the next one
        self.cursor = i
        return True

    def history_step(self, direction: int):
        """Replaces the text with the previous (-1) or next (+1) message sent; past the newest, the draft comes
        back."""
        pos = self.history_pos + direction
        if not 0 <= pos <= len(self.history):
            return
        if self.history_pos == len(self.history):
            self.draft = self.text
        self.history_pos = pos
        self.text = self.history[pos] if pos < len(self.history) else self.draft
        self.cursor = len(self.text)

# Control bytes -> key names
CONTROL = {0x0d: "enter", 0x0a: "newline", 0x7f: "backspace", 0x08: "backspace", 0x09: "tab", 0x01: "home",
           0x05: "end", 0x02: "left", 0x06: "right", 0x15: "kill-line", 0x0b: "kill-to-end", 0x17: "kill-word",
           0x04: "eof", 0x0c: "redraw"}

def csi_key(params: str, final: str) -> str:
    """The key an escape sequence ESC [ params final stands for."""
    if final in "ABCD":
        modified = params.endswith((";3", ";5", ";9"))  # Alt, Ctrl or Cmd held: by word
        return {"A": "up", "B": "down", "C": "word-right" if modified else "right",
                "D": "word-left" if modified else "left"}[final]
    if final in "HF":
        return "home" if final == "H" else "end"
    if final == "~":
        if params.startswith("27;") and params.endswith(";13"):  # xterm's modifyOtherKeys: Shift+Enter
            return "newline"
        return {"1": "home", "7": "home", "4": "end", "8": "end", "3": "delete",
                "200": "paste-start", "201": "paste-end"}.get(params, "ignore")
    if final == "u":  # the "CSI u" keyboard protocol (kitty, iTerm2, WezTerm when enabled)
        if params == "13":
            return "enter"
        if params.startswith("13;"):
            return "newline"  # Enter with a modifier: Shift+Enter
        if params.startswith("27"):
            return "esc"
    return "ignore"

class Keys:
    """Reads key presses from the terminal (in cbreak mode). next() gives ("text", str) for typed text, a key name
    ("enter", "left", "paste-start", ...), or None when no whole key has arrived yet."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = b""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def fill(self, timeout: float) -> bool:
        """Waits up to timeout seconds for more bytes. -> whether any came."""
        if select.select([self.fd], [], [], max(0.0, timeout))[0]:
            data = os.read(self.fd, 65536)
            if data:
                self.buf += data
                return True
        return False

    def more_waiting(self) -> bool:
        return bool(self.buf) or bool(select.select([self.fd], [], [], 0)[0])

    def next(self):
        if not self.buf:
            return None
        first = self.buf[0]
        if first == 0x1b:
            return self.escape()
        if first < 0x20 or first == 0x7f:
            self.buf = self.buf[1:]
            return CONTROL.get(first, "ignore")
        end = next((i for i, b in enumerate(self.buf) if b < 0x20 or b == 0x7f), len(self.buf))
        text, self.buf = self.buf[:end], self.buf[end:]
        decoded = self.decoder.decode(text)
        return ("text", decoded) if decoded else "ignore"

    def escape(self):
        """A key that starts with ESC: Esc itself, an arrow or function key (ESC [ or ESC O, then more), or
        Alt+key (ESC, then the key)."""
        if len(self.buf) == 1 and not self.fill(ESC_WAIT):
            self.buf = b""
            return "esc"
        second = self.buf[1]
        if second == ord("["):
            match = re.match(rb"\x1b\[([0-9;?]*)([\x40-\x7e])", self.buf)
            while match is None and len(self.buf) < 64 and re.fullmatch(rb"\x1b\[[0-9;?]*", self.buf):
                if not self.fill(ESC_WAIT):  # cut short: drop it
                    self.buf = b""
                    return "ignore"
                match = re.match(rb"\x1b\[([0-9;?]*)([\x40-\x7e])", self.buf)
            if match is None:
                self.buf = self.buf[2:]
                return "ignore"
            self.buf = self.buf[match.end():]
            return csi_key(match.group(1).decode(), chr(match.group(2)[0]))
        if second == ord("O"):
            if len(self.buf) < 3 and not self.fill(ESC_WAIT):
                self.buf = b""
                return "ignore"
            key, self.buf = chr(self.buf[2]), self.buf[3:]
            return {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}.get(key, "ignore")
        self.buf = self.buf[2:]
        if second in (0x0d, 0x0a):
            return "newline"    # Alt+Enter
        if second == 0x7f:
            return "kill-word"  # Alt+Backspace
        return {ord("b"): "word-left", ord("f"): "word-right"}.get(second, "ignore")  # Alt+B, Alt+F

class ConversationStream:
    """sys.stdout while the screen is open: everything printed goes into the conversation."""

    def __init__(self, screen: "ChatScreen"):
        self.screen = screen
        self.encoding = screen.out.encoding

    def write(self, text: str) -> int:
        if text:
            self.screen.conversation(text)
        return len(text)

    def flush(self):
        pass

    def isatty(self) -> bool:
        return True

class ChatScreen:
    """A context manager: entering takes over the terminal window, leaving gives it back as it was (also on
    Ctrl-C). The chat loop calls read() for each message, then begin(), token() for every generated token, and
    end() around each reply."""

    def __init__(self, title: str, engine):
        self.title, self.engine = title, engine
        self.out = sys.stdout  # the real terminal
        self.lock = threading.RLock()
        self.state = "idle"    # "idle", "reading" (the prompt's prefill) or "writing" (decoding)
        self.started = 0.0     # when the current phase began
        self.prompt_tokens = 0
        self.tokens = 0        # tokens generated in the current reply
        self.recent = deque(maxlen=SPEED_WINDOW)  # their arrival times
        self.last_reply = ""   # the last reply's speed, for the idle status line
        self.resize_at = None   # when the window last changed size, until that's been handled
        self.transcript = []    # everything written to the conversation, for redrawing it after a resize
        self.transcript_chars = 0
        self.stop = threading.Event()
        self.stopping = False   # Esc was pressed during this reply
        self.saved_tty = None   # the terminal's settings while the box or a reply has it in cbreak mode
        self.editor = Editor()
        self.editing = False    # the box has the keyboard
        self.input_rows = 1     # rows the box's text takes up (it grows while a long message is typed)
        self.box_scroll = 0     # the first of the message's rows the box shows, once it has more than fit
        self.conv_row = self.conv_col = 1  # where the conversation ends (known while editing; see find_cursor)

    @property
    def reserved(self) -> int:
        """Rows below the conversation: the status line, the box's two borders and its text."""
        return 3 + self.input_rows

    def max_input_rows(self) -> int:
        return max(1, min(MAX_INPUT_ROWS, self.height - 3 - MIN_CONVERSATION_ROWS))

    def box_width(self) -> int:
        """Columns for text in the box: │ › text │, with a space on each side of the text."""
        return max(1, self.width - 6)

    # Opening and closing

    def __enter__(self):
        self.width, self.height = shutil.get_terminal_size()
        # Clear the window and its scrollback (as `clear` does), hide the cursor (it only shows in the input box),
        # then start the conversation at the top, with the status line and the box at the bottom
        self.raw(f"{CSI}H{CSI}2J{CSI}3J{CSI}?25l")
        self.raw(f"{CSI}1;{self.height - self.reserved}r{CSI}1;1H{SAVE}")
        self.draw()
        self.old_winch = signal.signal(signal.SIGWINCH, self.on_resize)
        sys.stdout = ConversationStream(self)
        threading.Thread(target=self.refresh_loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.restore_keys()  # Ctrl-C mid-reply, or in the box
        with self.lock:
            sys.stdout = self.out
            signal.signal(signal.SIGWINCH, self.old_winch)
            # Turn bracketed paste off, drop the scroll region, show the cursor, and leave it under the box
            self.raw(f"{RESET}{CSI}?2004l{CSI}r{CSI}?25h{CSI}{self.height};1H\n")
            self.out.flush()
        return False

    def raw(self, text: str):
        self.out.write(text)

    def on_resize(self, signum, frame):
        self.resize_at = time.monotonic()  # handled once the size has settled (see apply_resize)

    def resize_settled(self) -> bool:
        return self.resize_at is not None and time.monotonic() - self.resize_at >= RESIZE_SETTLE

    def apply_resize(self) -> bool:
        """Once the window has settled at a new size: clears the screen and the scrollback, and prints the whole
        conversation again at the new width, the status line and box below it. -> whether it did."""
        if not self.resize_settled():
            return False
        self.resize_at = None
        self.width, self.height = shutil.get_terminal_size()
        self.input_rows = min(self.input_rows, self.max_input_rows())
        bottom = self.height - self.reserved
        self.raw(f"{RESET}{CSI}r{CSI}H{CSI}2J{CSI}3J")  # drop the scroll region, clear the screen and the scrollback
        self.raw(f"{CSI}1;{bottom}r{CSI}1;1H")
        self.raw("".join(self.transcript))
        self.raw(SAVE)  # the conversation goes on where the transcript ends, in the style it ended in
        self.conv_row, self.conv_col = bottom, 1  # until find_cursor says where that is
        self.draw()
        return True

    # The conversation

    def conversation(self, text: str):
        with self.lock:
            self.transcript.append(text)
            self.transcript_chars += len(text)
            if self.transcript_chars > TRANSCRIPT_LIMIT:  # keep the newest part, from a line's start
                kept = "".join(self.transcript)[-TRANSCRIPT_LIMIT // 2:]
                kept = RESET + kept[kept.find("\n") + 1:]
                self.transcript, self.transcript_chars = [kept], len(kept)
            if self.resize_at is None:  # mid-resize, it shows when the screen is redrawn
                self.raw(RESTORE + text + SAVE)
                self.out.flush()

    def say(self, text: str):
        self.conversation(text + "\n")

    # The input box

    def read(self) -> str:
        """Waits for the next message (or command) in the box. -> its text, stripped. Ctrl-D in an empty box
        raises EOFError."""
        fd = sys.stdin.fileno()
        with self.lock:
            self.state = "idle"
            self.apply_resize()
        self.saved_tty = termios.tcgetattr(fd)
        try:
            # cbreak: each key arrives as it's pressed, unechoed; Ctrl-C still interrupts. Without ICRNL, Enter
            # (CR) and Ctrl-J (LF) stay apart; without IXON, Ctrl-S and Ctrl-Q reach the box
            tty.setcbreak(fd)
            attrs = termios.tcgetattr(fd)
            attrs[0] &= ~(termios.ICRNL | termios.IXON)
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            keys = Keys(fd)
            with self.lock:
                self.raw(CSI + "?2004h")  # bracketed paste: pasted text comes marked, so its newlines don't send it
                self.find_cursor(keys)
                self.editing = True
                self.layout_box()
            return self.edit(keys).strip()
        finally:
            with self.lock:
                self.editing = False
                self.box_scroll = 0
                self.raw(f"{CSI}?2004l{CSI}?25l")
                self.restore_keys()
                self.set_box_rows(1)
                self.draw()

    def find_cursor(self, keys: Keys):
        """Asks the terminal where the conversation's cursor is (DSR), so the box knows how far it can grow before
        it has to push the conversation up. Keys typed meanwhile are kept."""
        self.raw(RESTORE + CSI + "6n")
        self.out.flush()
        deadline = time.monotonic() + 0.3
        while True:
            match = DSR_REPLY.search(keys.buf)
            if match:
                keys.buf = keys.buf[:match.start()] + keys.buf[match.end():]
                self.conv_row, self.conv_col = int(match.group(1)), int(match.group(2))
                return
            if not keys.fill(deadline - time.monotonic()):
                break
        # No answer: assume the conversation reaches the box, so growing it always pushes the conversation up
        self.conv_row, self.conv_col = self.height - self.reserved, 1

    def edit(self, keys: Keys) -> str:
        """Edits the message as keys arrive, until Enter. -> the text."""
        ed = self.editor
        pasting = after_cr = False
        while True:
            if self.resize_settled():
                with self.lock:
                    if self.apply_resize():
                        self.find_cursor(keys)  # where the redrawn conversation ends
                        self.layout_box()
            key = keys.next()
            if key is None:
                keys.fill(0.1)
                continue
            if isinstance(key, tuple):  # typed or pasted text
                ed.insert(key[1])
            elif pasting:  # inside a paste, Enter and Ctrl-J are its newlines (CR LF counts once)
                if key == "paste-end":
                    pasting = False
                elif key == "enter" or (key == "newline" and not after_cr):
                    ed.insert("\n")
                elif key == "tab":
                    ed.insert(TAB)
            elif key == "enter":
                if keys.more_waiting():
                    ed.insert("\n")  # more text right behind it: a paste the terminal didn't mark
                else:
                    return ed.submit()
            elif key == "paste-start":
                pasting = True
            elif key == "newline":
                ed.insert("\n")
            elif key == "tab":
                ed.insert(TAB)
            elif key == "backspace":
                ed.backspace()
            elif key == "delete":
                ed.delete()
            elif key == "eof":
                if not ed.text:
                    raise EOFError
                ed.delete()
            elif key == "esc":
                ed.clear()
            elif key == "left":
                ed.move(-1)
            elif key == "right":
                ed.move(1)
            elif key == "word-left":
                ed.cursor = ed.word_left()
            elif key == "word-right":
                ed.cursor = ed.word_right()
            elif key == "home":
                ed.cursor = ed.line_start()
            elif key == "end":
                ed.cursor = ed.line_end()
            elif key in ("up", "down"):
                direction = -1 if key == "up" else 1
                if not ed.vertical(self.box_width(), direction):
                    ed.history_step(direction)
            elif key == "kill-line":
                ed.cut(ed.line_start(), ed.cursor)
            elif key == "kill-to-end":
                ed.cut(ed.cursor, ed.line_end())
            elif key == "kill-word":
                ed.cut(ed.word_left(), ed.cursor)
            after_cr = pasting and key == "enter"
            if not keys.buf and self.resize_at is None:  # once a burst of keys (a paste) has been handled
                with self.lock:
                    self.layout_box()

    def layout_box(self):
        """Sizes the box to the message (scrolling it once it's taller than the box can grow) and draws it."""
        rows = wrap(self.editor.text, self.box_width())
        row, _ = locate(rows, self.editor.text, self.editor.cursor)
        n = max(1, min(len(rows), self.max_input_rows()))
        self.box_scroll = max(0, min(self.box_scroll, row, len(rows) - n))
        if row >= self.box_scroll + n:
            self.box_scroll = row - n + 1
        self.set_box_rows(n)
        self.draw()

    def set_box_rows(self, n: int):
        """Gives the box n rows of text. Growing it first scrolls the conversation up by as much as the box would
        cover of it (into the terminal's scrollback); shrinking it gives the rows back to the conversation."""
        if n == self.input_rows:
            return
        old_bottom = self.height - self.reserved
        new_bottom = self.height - 3 - n
        if n > self.input_rows:
            covered = self.conv_row - new_bottom
            if covered > 0:
                self.raw(f"{CSI}{old_bottom};1H" + "\n" * covered)  # line feeds at the region's bottom scroll it
                self.conv_row -= covered
        else:
            for row in range(old_bottom + 1, new_bottom + 1):
                self.raw(f"{CSI}{row};1H{CSI}2K")
        self.raw(f"{CSI}1;{new_bottom}r{CSI}{self.conv_row};{self.conv_col}H{SAVE}")
        self.input_rows = n

    # The reply

    def begin(self, message: str, prompt_tokens: int):
        """A message is about to be sent: echoes it into the conversation, starts timing its prompt, and starts
        listening for Esc."""
        self.stopping = False
        self.watch_keys()
        with self.lock:
            self.conversation(f"\n{BOLD}› {message.replace(chr(10), chr(10) + '  ')}{RESET}\n\n")
            self.state, self.started, self.prompt_tokens = "reading", time.perf_counter(), prompt_tokens
            self.tokens = 0
            self.recent.clear()
            self.draw()

    def token(self) -> bool:
        """Called for every generated token. -> True once Esc has been pressed: stop the reply here."""
        now = time.perf_counter()
        with self.lock:
            if self.state == "reading":
                self.state, self.started = "writing", now
            self.tokens += 1
            self.recent.append(now)
        if not self.stopping and self.esc_pressed():
            self.stopping = True
            with self.lock:
                self.draw()
        return self.stopping

    def end(self, summary: str):
        """summary goes into the conversation; the idle status line keeps just the reply's speed."""
        self.restore_keys()
        if self.stopping:
            summary = "stopped · " + summary
        with self.lock:
            if self.state == "writing":
                _, average = self.speeds()
                self.last_reply = f"{average:.1f} tok/s"
            self.state = "idle"
            self.conversation(f"\n{DIM}[{summary}]{RESET}\n")
            self.draw()

    # Esc during a reply

    def watch_keys(self):
        """cbreak mode for the reply: each key press can be read as it comes, and isn't echoed into the screen.
        Ctrl-C still interrupts."""
        fd = sys.stdin.fileno()
        self.saved_tty = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def restore_keys(self):
        """Back to the terminal's own settings, dropping any keys pressed but not read."""
        if self.saved_tty is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, self.saved_tty)
            self.saved_tty = None

    def esc_pressed(self) -> bool:
        """Reads the keys pressed since the last token, without waiting. Esc alone sends one ESC byte; arrow and
        function keys send ESC [ or ESC O and more, which don't count. Other keys are ignored."""
        fd = sys.stdin.fileno()
        pressed = False
        while select.select([fd], [], [], 0)[0]:
            keys = os.read(fd, 1024)
            if not keys:
                break
            for i, byte in enumerate(keys):
                if byte == 0x1b and keys[i + 1:i + 2] not in (b"[", b"O"):
                    pressed = True
        return pressed

    # The bottom rows

    def refresh_loop(self):
        """Redraws the status line while the model works, so the timer moves even while a prompt is read (when
        no token arrives), and handles a resize once it has settled (the box handles its own while editing)."""
        last_draw = 0.0
        while not self.stop.wait(0.05):
            with self.lock:
                if not self.editing and self.apply_resize():
                    last_draw = time.monotonic()
                elif self.state != "idle" and time.monotonic() - last_draw >= REFRESH_SECONDS:
                    self.draw()
                    last_draw = time.monotonic()

    def speeds(self) -> tuple:
        """-> (tokens per second over the last SPEED_WINDOW tokens, over the whole reply so far)."""
        now = time.perf_counter()
        live = (len(self.recent) - 1) / (self.recent[-1] - self.recent[0]) if len(self.recent) > 1 else 0.0
        if self.recent and now - self.recent[-1] > 2 * REFRESH_SECONDS and live:
            # No new token for a while (a slow token, mostly SSD reads): let the live speed fall meanwhile
            live = min(live, len(self.recent) / (now - self.recent[0]))
        elapsed = now - self.started
        average = (self.tokens - 1) / elapsed if self.tokens > 1 and elapsed > 0 else 0.0
        return live, average

    def status(self) -> tuple:
        """-> (left part as [(text, style)], right part as plain text)."""
        context = f"context {self.engine.position:,}"
        if self.state == "reading":
            elapsed = time.perf_counter() - self.started
            return ([(f" reading your message: {self.prompt_tokens:,} tokens · {elapsed:.1f}s", "")],
                    "Esc stops · Ctrl-C quits")
        if self.state == "writing":
            live, average = self.speeds()
            elapsed = time.perf_counter() - self.started
            left = [(f" {live:5.1f} tok/s", BOLD + GREEN),
                    (f"  (avg {average:.1f}) · {self.tokens:,} tokens · {elapsed:.1f}s · {context}", "")]
            if self.stopping:
                return [(" stopping after this token…", BOLD)] + left[1:], "Ctrl-C quits"
            return left, "Esc stops · Ctrl-C quits"
        thinking = "on" if self.engine.think else "off"
        left = [(f" {self.title}", BOLD), (f" · {context} · thinking {thinking}", "")]
        if self.last_reply:
            left.append((f" · last reply {self.last_reply}", ""))
        return left, "/think /reset /exit"

    def status_line(self) -> str:
        w = self.width
        left, right = self.status()
        if sum(len(text) for text, _ in left) + len(right) + 3 > w:
            right = ""  # too narrow for both: the hint goes first
        room = w - 1 - (len(right) + 2 if right else 0)
        line, used = "", 0
        for text, style in left:  # cut the left part to fit, keeping the right part
            text = text[:max(0, room - used)]
            line += f"{style}{text}{RESET}" if style else text
            used += len(text)
        if right and room >= 0:
            line += " " * (room - used + 1) + f"{DIM}{right}{RESET}" + " "
        return line

    def draw(self):
        """Draws the status line and the box, then puts the cursor back: into the box while editing, otherwise
        where the conversation left off. Not while a resize is settling: apply_resize redraws everything."""
        if self.resize_at is not None:
            return
        w, n = self.width, self.input_rows
        top = self.height - self.reserved + 1  # the status line's row
        inner, avail = max(0, w - 2), self.box_width()
        text = self.editor.text if self.editing else ""
        rows = wrap(text, avail)
        first = self.box_scroll if self.editing else 0
        above, below = first, max(0, len(rows) - first - n)
        parts = [f"{RESET}{CSI}{top};1H{CSI}2K{self.status_line()}"]
        label = f" ↑ {above} more " if above else ""
        parts.append(f"{CSI}{top + 1};1H{CSI}2K{DIM}╭{label}{'─' * max(0, inner - len(label))}╮{RESET}")
        for i in range(n):
            start, end = rows[first + i] if first + i < len(rows) else (0, 0)
            segment = text[start:end]
            if text_width(segment) > avail:
                segment = segment.rstrip(" ")  # a row that breaks at a space: the space would cover the border
            mark = f"{BOLD}›{RESET}" if first + i == 0 else " "
            pad = " " * max(0, avail - text_width(segment))
            parts.append(f"{CSI}{top + 2 + i};1H{CSI}2K{DIM}│{RESET} {mark} {segment}{pad} {DIM}│{RESET}")
        label = f" ↓ {below} more " if below else ""
        hint = BOX_HINT + "─" if self.editing else ""
        if len(label) + len(hint) > inner:
            hint = ""
        fill = "─" * max(0, inner - len(label) - len(hint))
        parts.append(f"{CSI}{top + 2 + n};1H{CSI}2K{DIM}╰{label}{fill}{hint}╯{RESET}")
        if self.editing:
            row, column = locate(rows, text, self.editor.cursor)
            parts.append(f"{CSI}{top + 2 + row - first};{5 + column}H{CSI}?25h")
        else:
            parts.append(RESTORE)
        self.raw("".join(parts))
        self.out.flush()
