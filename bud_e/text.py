"""Turning a token stream into things worth saying out loud."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
_MARKDOWN = re.compile(r"(\*\*|\*|__|`+|^#+\s*|^\s*[-•]\s+|^\s*\d+\.\s+)",
                       re.M)
_URL = re.compile(r"https?://\S+")


class ThinkFilter:
    """Drops <think>...</think> blocks from a stream, even when the tags are
    split across chunks. Some servers inline reasoning into content."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    def feed(self, piece: str) -> str:
        self.buf += piece
        out = []
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if not self.inside:
                    out.append(self.buf[:i])
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            # Keep a tail that could be the start of a split tag.
            keep = next((k for k in range(len(tag) - 1, 0, -1)
                         if self.buf.endswith(tag[:k])), 0)
            ready = self.buf[:len(self.buf) - keep]
            self.buf = self.buf[len(self.buf) - keep:]
            if not self.inside:
                out.append(ready)
            return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


class Sentences:
    """Collects streamed text and hands back whole sentences as they finish,
    so speech can start at the first full stop."""

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        parts = _SENTENCE_END.split(self.buf)
        self.buf = parts.pop()
        return [s for s in (speakable(p) for p in parts) if s]

    def flush(self) -> list[str]:
        s, self.buf = speakable(self.buf), ""
        return [s] if s else []


def speakable(s: str) -> str:
    """Strip what a speaker can't say: markdown, URLs, stray whitespace."""
    s = _URL.sub("", s)
    s = _MARKDOWN.sub("", s)
    return " ".join(s.split())
