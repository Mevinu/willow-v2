"""Shows the Markdown a model writes as styled terminal text, as it streams in: the markers go, their meaning
stays. Used by runner.ReplyPrinter whenever stdout is a terminal (piped output keeps the Markdown as written).

    # Heading            -> bold (and underlined for # and ##)
    **bold**, *italic*   -> bold, italic
    `code`               -> in the code color
    ```lang ... ```      -> the code as it is, in the code color, between dim rules (copyable: no prefix)
    - item, * item       -> • item
    > quote              -> │ quote, dim bar
    ---                  -> a dim rule
    | a | b |            -> columns lined up (shown once the table is complete)
    [text](url)          -> text (url)
    \\*                  -> *

Text arrives a few characters at a time and can split any marker, so it is read one character at a time. The
start of each line is held back until it's clear what kind of line it is (a heading, a list item, a rule, a fence,
a table row); a run of * waits for the character after it, and a [ waits to see whether a link follows. Styles
never carry past the end of a line, so an unclosed ** can't turn the rest of the reply bold."""
import re
import shutil
import unicodedata

CSI = "\033["
RESET, BOLD, DIM, ITALIC, UNDERLINE, CODE = (CSI + c + "m" for c in ("0", "1", "2", "3", "4", "36"))
PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
LINK_LIMIT = 500  # a [ with no ](...) within this many characters is just a bracket
SEPARATOR_CELL = re.compile(r":?-{1,}:?")

def text_width(text: str) -> int:
    return sum(0 if unicodedata.combining(ch) else 2 if unicodedata.east_asian_width(ch) in "WF" else 1
               for ch in text)

def plain_cell(cell: str) -> str:
    """A table cell's text without its inline markers (cells are lined up by their visible width)."""
    cell = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", r"\1", cell)
    return re.sub(r"\*\*|__|`|(?<!\w)\*(?=\S)|(?<=\S)\*(?!\w)", "", cell).replace("\\|", "|")

class MarkdownStream:
    """feed(text) -> what to print for it; finish() -> what is still held back, at the end of the reply.
    base is the style everything is shown in (DIM for the model's thinking)."""

    def __init__(self, base: str = ""):
        self.base = base
        self.at_line_start = True
        self.head = ""          # the start of the current line, held until its kind is known
        self.line_kind = None   # the rest of the line: "text" (with inline Markdown) or "code" (as it is)
        self.line_style = ""    # the whole line's style (a heading's)
        self.fence = None       # the ``` or ~~~ that opened the code block we're in, if any
        self.table = []         # the rows of the table being collected
        self.bold = self.italic = self.code = False
        self.stars = 0          # a run of * waiting for the next character
        self.escape = False     # after a backslash
        self.link = None        # since a [: [text, stage ("text", "paren" or "url"), url]
        self.prev = " "         # the last character shown (for whether a * can close)

    # Feeding text

    def feed(self, text: str) -> str:
        return "".join(self.char(ch) for ch in text)

    def finish(self) -> str:
        out = ""
        if self.at_line_start:
            if self.head:
                out += self.whole_line(self.head).removesuffix("\n")
            self.head = ""
        else:
            out += self.close_inline("\n")
        if self.table:
            out += self.render_table().removesuffix("\n")
        if self.fence is not None:
            out += "\n" + self.rule("")
            self.fence = None
        return out + RESET + self.base

    def char(self, ch: str) -> str:
        if self.at_line_start:
            if ch == "\n":
                line, self.head = self.head, ""
                return self.whole_line(line)
            self.head += ch
            kind = self.classify(self.head, done=False)
            return "" if kind is None else self.start_line(kind)
        if ch == "\n":
            return self.end_line()
        if self.line_kind == "code":
            return ch
        return self.inline(ch)

    # Lines

    def classify(self, head: str, done: bool):
        """What kind of line head starts. -> (kind, detail) or None while it can't be told yet (done: head is
        the whole line)."""
        s = head.lstrip(" ")
        indent = len(head) - len(s)
        if self.fence is not None:
            if s.startswith(self.fence):
                return ("fence-close", None) if done else None
            if self.fence.startswith(s) and not done:
                return None
            return ("code", None)
        if not s:
            return ("text", None) if done else None
        first = s[0]
        if first in "`~":
            if s.startswith(first * 3):
                return ("fence-open", s[3:].strip()) if done else None
            if set(s) == {first} and not done:
                return None
            return ("text", None)
        if first == "|":
            return ("table", None) if done else None
        if first in "-*_" and set(s) <= {first, " "}:  # only markers so far: a rule, or the start of something
            if not done:
                return None
            if s.count(first) >= 3:
                return ("rule", None)
            return ("text", None)
        if first == "+" and len(s) == 1 and not done:
            return None  # a bullet if a space follows
        if first in "-*+" and len(s) >= 2 and s[1] == " ":
            return ("bullet", indent)
        if first == "#":
            level = len(s) - len(s.lstrip("#"))
            if level == len(s) and not done:
                return None
            if level <= 6 and len(s) > level and s[level] == " ":
                return ("heading", level)
            return ("text", None)
        if first == ">":
            if len(s) == 1 and not done:
                return None
            return ("quote", indent)
        return ("text", None)

    def start_line(self, kind) -> str:
        """The line's kind is known: shows its start, then goes on reading it a character at a time."""
        name, detail = kind
        head, self.head = self.head, ""
        self.at_line_start = False
        out = self.flush_table() if name != "table" else ""
        s = head.lstrip(" ")
        if name == "code":
            self.line_kind = "code"
            return out + CODE + head
        self.line_kind = "text"
        if name == "bullet":
            out += " " * detail + "• "
            rest = s[2:]
        elif name == "heading":
            self.line_style = BOLD + (UNDERLINE if detail <= 2 else "")
            out += self.restyle()
            rest = s[detail + 1:].lstrip(" ")
        elif name == "quote":
            out += " " * detail + DIM + "│ " + RESET + self.base
            rest = s[1:].lstrip(" ")
        else:
            rest = head
        return out + "".join(self.inline(ch) for ch in rest)

    def whole_line(self, line: str) -> str:
        """A line that ended before its kind was known (or that needs all of it: rules, fences, table rows)."""
        name, detail = self.classify(line, done=True)
        if name == "table":
            self.table.append(line)
            return ""
        out = self.flush_table()
        if name == "rule":
            return out + self.rule("") + "\n"
        if name == "fence-open":
            self.fence = line.lstrip(" ")[:3]
            return out + self.rule(detail) + "\n"
        if name == "fence-close":
            self.fence = None
            return out + self.rule("") + "\n"
        self.head = line
        return out + self.start_line((name, detail)) + self.end_line()

    def end_line(self) -> str:
        out = self.close_inline("\n")
        self.at_line_start, self.line_kind = True, None
        return out + "\n"

    def close_inline(self, next_char: str) -> str:
        """Ends the line's inline state: a waiting * or [ shows as it is, and styles stop here."""
        out = self.resolve_stars(next_char) if self.stars else ""
        if self.link is not None:
            out += self.unlink()
        self.escape = False
        styled = self.bold or self.italic or self.code or self.line_style or self.line_kind == "code"
        self.bold = self.italic = self.code = False
        self.line_style = ""
        self.prev = " "
        return out + (RESET + self.base if styled else "")

    def rule(self, label: str) -> str:
        width = min(shutil.get_terminal_size().columns - 1, 60)
        text = f"─── {label} " if label else ""
        return DIM + text + "─" * max(3, width - text_width(text)) + RESET + self.base

    # Inline Markdown

    def restyle(self) -> str:
        return (RESET + self.base + self.line_style + (BOLD if self.bold else "") + (ITALIC if self.italic else "")
                + (CODE if self.code else ""))

    def show(self, ch: str) -> str:
        self.prev = ch
        return ch

    def inline(self, ch: str) -> str:
        if self.link is not None:
            return self.link_char(ch)
        out = ""
        if self.stars:
            if ch == "*" and not self.code:
                self.stars += 1
                return ""
            out = self.resolve_stars(ch)
        if self.escape:
            self.escape = False
            return out + (self.show(ch) if ch in PUNCTUATION else self.show("\\") + self.show(ch))
        if self.code:
            if ch == "`":
                self.code = False
                return out + self.restyle()
            return out + self.show(ch)
        if ch == "\\":
            self.escape = True
            return out
        if ch == "*":
            self.stars = 1
            return out
        if ch == "`":
            self.code = True
            return out + self.restyle()
        if ch == "[":
            self.link = ["", "text", ""]
            return out
        return out + self.show(ch)

    def resolve_stars(self, next_char: str) -> str:
        """A run of * ends before next_char: ** turns bold on or off, * italic, where Markdown would (a * can open
        before a non-space and close after one); any left over show as they are."""
        count, self.stars = self.stars, 0
        can_open, can_close = not next_char.isspace(), not self.prev.isspace()
        changed = False
        if count >= 2 and ((self.bold and can_close) or (not self.bold and can_open)):
            self.bold, count, changed = not self.bold, count - 2, True
        if count >= 1 and ((self.italic and can_close) or (not self.italic and can_open)):
            self.italic, count, changed = not self.italic, count - 1, True
        return (self.restyle() if changed else "") + "".join(self.show("*") for _ in range(count))

    def link_char(self, ch: str) -> str:
        """[text](url) shows as text (url); anything else that started with [ shows as it was written."""
        text, stage, url = self.link
        if stage == "text":
            if ch == "]":
                self.link[1] = "paren"
                return ""
            if ch == "[" or len(text) >= LINK_LIMIT:
                return self.unlink() + self.inline(ch)
            self.link[0] += ch
            return ""
        if stage == "paren":
            if ch == "(":
                self.link[1] = "url"
                return ""
            return self.unlink() + self.inline(ch)
        if ch == ")":
            self.link = None
            label = plain_cell(text)
            shown = url if label in ("", url) else f"{label} {DIM}({url}){self.restyle()}"
            self.prev = ")"
            return shown
        if ch == " " or len(url) >= LINK_LIMIT:
            return self.unlink() + self.inline(ch)
        self.link[2] += ch
        return ""

    def unlink(self) -> str:
        """Shows a [ that turned out not to start a link, and what came after it, as written."""
        text, stage, url = self.link
        self.link = None
        raw = "[" + text + ("]" if stage != "text" else "") + ("(" + url if stage == "url" else "")
        return "".join(self.show(c) for c in raw)

    # Tables

    def flush_table(self) -> str:
        return self.render_table() if self.table else ""

    def render_table(self) -> str:
        rows = []
        for line in self.table:
            body = line.strip()
            body = body[1:] if body.startswith("|") else body
            body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
            rows.append([plain_cell(cell.strip()) for cell in re.split(r"(?<!\\)\|", body)])
        self.table = []
        is_separator = [all(SEPARATOR_CELL.fullmatch(c) for c in row if c) and any(row) for row in rows]
        header = 1 if len(rows) > 1 and is_separator[1] else 0
        rows = [row for row, sep in zip(rows, is_separator) if not sep]
        columns = max(len(row) for row in rows) if rows else 0
        widths = [max((text_width(row[i]) for row in rows if i < len(row)), default=0) for i in range(columns)]
        out = []
        for i, row in enumerate(rows):
            cells = [row[j] if j < len(row) else "" for j in range(columns)]
            line = "  ".join(c + " " * (widths[j] - text_width(c)) for j, c in enumerate(cells)).rstrip()
            out.append(BOLD + line + RESET + self.base if i < header else line)
            if i == header - 1:
                out.append(DIM + "─" * (sum(widths) + 2 * (columns - 1)) + RESET + self.base)
        return "\n".join(out) + "\n"
