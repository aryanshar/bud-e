"""Fast tier: a small model answering directly, no Hermes, no tools.

The whole point is the prompt is ~200 tokens instead of ~13,000, so there is
no prefill to pay and replies start almost immediately. It streams sentence
by sentence so speech begins before generation finishes.
"""

import re
from collections.abc import Iterator

from bud_e import config as C
from bud_e.llm import local
from bud_e.persona import fast_prompt

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def system_prompt() -> str:
    return fast_prompt(C.ADDRESS, C.ROUTE_TOKEN, C.PERSONA)


class Fast:
    """Holds a short rolling history so follow-ups make sense."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def note(self, user: str, reply: str) -> None:
        """Record a turn the agent answered, so follow-ups make sense here."""
        self.history += [{"role": "user", "content": user},
                         {"role": "assistant", "content": reply}]
        self._trim()

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
            temperature=0.8, top_p=0.95, max_tokens=C.FAST_MAX_TOKENS)

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
