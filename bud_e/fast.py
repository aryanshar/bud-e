"""Fast tier: a small model answering directly, no Hermes, no tools.

The whole point is the prompt is ~200 tokens instead of ~13,000, so there is
no prefill to pay and replies start almost immediately. It streams sentence
by sentence so speech begins before generation finishes.
"""

import re
from collections.abc import Iterator

from bud_e import config as C
from bud_e.llm import local

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

BASE = f"""You are Bud-E, {{addr}}'s voice assistant. You are spoken aloud, so:

- One or two sentences. Never more unless asked to go on.
- No markdown, no lists, no headings, no URLs, no emoji.
- Numbers as spoken: "about forty five thousand", not "44,873".
- Answer first. Never restate the question.
- Dry, calm, faintly amused. Address them as {{addr}} occasionally, not every line.
- You are talking, not writing. Contractions, plain words, no preamble.

You have no tools, no memory, no internet, and no access to files or calendars.
If the request needs any of those, or needs real reasoning, multi-step work, or
anything you would have to guess at, reply with exactly {C.ROUTE_TOKEN} and
nothing else. A colleague with tools will take it. Do not apologise, do not
explain, do not attempt it.

Answer directly when it is conversation, general knowledge you are confident
about, or anything about the moment you are both in."""


def system_prompt() -> str:
    s = BASE.replace("{addr}", C.ADDRESS)
    return s + ("\n\n" + C.PERSONA if C.PERSONA else "")


class Fast:
    """Holds a short rolling history so follow-ups make sense."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def _trim(self) -> None:
        keep = C.FAST_HISTORY * 2
        if len(self.history) > keep:
            self.history = self.history[-keep:]

    def turn(self, text: str) -> Iterator[tuple[str, str]]:
        """Yields ('say', sentence) as they complete, or ('route', '') once
        if the model declines. Caller speaks each sentence immediately."""
        msgs = [{"role": "system", "content": system_prompt()},
                *self.history,
                {"role": "user", "content": text}]

        stream = local().chat.completions.create(
            model=C.FAST_MODEL, messages=msgs, stream=True,
            temperature=0.6, max_tokens=C.FAST_MAX_TOKENS)

        buf, full, decided = "", "", False
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content or ""
            if not piece:
                continue
            buf += piece
            full += piece

            # Catch the route signal before speaking a word of it.
            if not decided:
                head = full.lstrip()[:len(C.ROUTE_TOKEN)]
                if head.upper() == C.ROUTE_TOKEN and len(full.lstrip()) >= len(C.ROUTE_TOKEN):
                    yield ("route", "")
                    return
                if len(full.strip()) >= len(C.ROUTE_TOKEN):
                    decided = True

            parts = SENTENCE_END.split(buf)
            if len(parts) > 1:
                for s in parts[:-1]:
                    if s.strip():
                        yield ("say", s.strip())
                buf = parts[-1]

        if buf.strip():
            yield ("say", buf.strip())

        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": full.strip()}]
        self._trim()
