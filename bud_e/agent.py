"""One model, one loop: stream the reply, run any tool calls, repeat.

Replies are yielded sentence by sentence so speech can start at the first
full stop. Pass a threading.Event as `cancel` and set it to stop mid-reply;
an interrupted turn is not kept in history.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import datetime

from bud_e.config import Config
from bud_e.llm import LLM
from bud_e.prompt import system_prompt
from bud_e.text import Sentences, ThinkFilter
from bud_e.tools import Toolbox, memory_snapshot

# Events yielded by Agent.turn
SAY, TOOL = "say", "tool"


def stamp() -> str:
    return datetime.now().strftime("%a %-d %b %Y, %H:%M")


class Agent:
    def __init__(self, cfg: Config, llm: LLM | None = None,
                 tools: Toolbox | None = None) -> None:
        self.cfg = cfg
        self.llm = llm or LLM(cfg)
        self.tools = tools or Toolbox(cfg)
        self.system = {"role": "system", "content": system_prompt(
            cfg, self.tools.names, memory_snapshot(cfg))}
        self.turns: list[list[dict]] = []   # each turn: user, tools, reply

    def reset(self) -> None:
        self.turns.clear()

    def messages(self, current: list[dict]) -> list[dict]:
        return [self.system, *(m for t in self.turns for m in t), *current]

    def warm(self) -> None:
        """Prefill the system prompt and tool schemas so the first real turn
        hits the server's prefix cache."""
        list(self.llm.stream([self.system, {"role": "user", "content": "hi"}],
                             self.tools.schemas(), max_tokens=1))

    def turn(self, text: str, cancel: threading.Event | None = None
             ) -> Iterator[tuple[str, str]]:
        """Yields (SAY, sentence) and (TOOL, name)."""
        cancel = cancel or threading.Event()
        current = [{"role": "user", "content": f"[{stamp()}] {text}"}]

        for _ in range(self.cfg.assistant.max_tool_steps):
            think, sentences = ThinkFilter(), Sentences()
            reply, calls = "", []
            for kind, value in self.llm.stream(self.messages(current),
                                               self.tools.schemas() or None):
                if cancel.is_set():
                    return
                if kind == "calls":
                    calls = value
                    break
                visible = think.feed(value)
                reply += visible
                for s in sentences.feed(visible):
                    yield SAY, s
            tail = think.flush()
            reply += tail
            for s in sentences.feed(tail) + sentences.flush():
                yield SAY, s
            if cancel.is_set():
                return

            if not calls:
                current.append({"role": "assistant", "content": reply.strip()})
                self._keep(current)
                return

            current.append({"role": "assistant", "content": reply or None,
                            "tool_calls": [c.as_message() for c in calls]})
            for c in calls:
                if cancel.is_set():
                    return
                yield TOOL, c.name
                current.append({"role": "tool", "tool_call_id": c.id,
                                "content": self.tools.run(c.name, c.arguments)})

        yield SAY, "I went round in circles on that one. Try asking another way."

    def _keep(self, turn: list[dict]) -> None:
        self.turns.append(turn)
        # Trim in big steps: every trim changes the prefix and costs a
        # fresh prefill, so do it rarely.
        limit = self.cfg.assistant.history_turns
        if len(self.turns) > limit:
            self.turns = self.turns[-max(1, limit // 2):]
