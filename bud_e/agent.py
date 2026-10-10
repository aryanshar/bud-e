"""The 27B, called directly: streamed, native tool calls, no Hermes.

Why this replaced `hermes chat --oneshot` on the voice path:
  - one process, no per-turn spawn
  - a ~1.5k token system prompt instead of ~13k, and it never changes, so the
    server's prefix cache hits on every turn (the date rides the user message)
  - replies stream sentence by sentence, so speech starts at the first full stop
  - depth turns switch thinking on instead of declining
"""

import re
from collections.abc import Callable, Iterator
from datetime import datetime

from bud_e import config as C
from bud_e import tools as T
from bud_e.llm import local, sampling
from bud_e.persona import agent_prompt

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
MARKDOWN = re.compile(r"[*#`_>|]+")

TOOLS_NOTE = """Tools:
- Use a tool whenever the answer depends on his calendar, reminders, notes, \
files, memory, the weather or the web. Never invent what they contain.
- Every message from him starts with the current date and time in brackets. \
Resolve words like tomorrow or Tuesday at four against it and pass absolute \
YYYY-MM-DD or YYYY-MM-DDTHH:MM values to tools.
- When he says something durable about his life, work or preferences, call \
remember. Don't announce it beyond a word like "noted".
- For questions about past conversations or his preferences, call recall first.
- A morning brief is today's calendar, reminders due today and the weather, \
in three sentences.
- If delegate is available, use it only for long multi-step jobs he's happy \
to hear back on later, and say you've started it.
- After a tool call, answer in a sentence or two. Never narrate the tool.
- If a tool fails, say so plainly in one sentence."""


def system_prompt() -> str:
    return agent_prompt(C.ADDRESS, TOOLS_NOTE, C.PERSONA)


def _clean(s: str) -> str:
    return MARKDOWN.sub("", s).strip()


class Agent:
    """Holds whole turns (user, tool exchanges, reply) so follow-ups like
    'and the one after that?' still see the calendar result."""

    def __init__(self) -> None:
        self.turns: list[list[dict]] = []
        self.tools = T.schemas()
        self.system = {"role": "system", "content": system_prompt()}

    def reset(self) -> None:
        self.turns.clear()

    def note(self, user: str, reply: str) -> None:
        """Record a turn the fast tier answered, so the conversation is one
        conversation whichever model spoke."""
        stamp = datetime.now().strftime("%a %-d %b %Y, %H:%M")
        self.turns.append([{"role": "user", "content": f"[{stamp}] {user}"},
                           {"role": "assistant", "content": reply}])
        self._trim()

    def _history(self) -> list[dict]:
        return [m for t in self.turns for m in t]

    def _trim(self) -> None:
        # Drop in chunks, not one turn at a time: every trim invalidates the
        # cached prefix, so do it rarely.
        if len(self.turns) > C.AGENT_HISTORY:
            self.turns = self.turns[-(C.AGENT_HISTORY // 2):]

    def warm(self) -> int:
        """Prefill the system prompt and tool schemas once at startup.
        Returns the prompt token count, which must clear oMLX's 2048-token
        block for the prefix to be cached on hybrid Qwen models."""
        r = local().chat.completions.create(
            model=C.LOCAL_MODEL, tools=self.tools,
            messages=[self.system, {"role": "user", "content": "hi"}],
            **sampling(False, 1))
        return getattr(r.usage, "prompt_tokens", 0) if r.usage else 0

    def turn(self, text: str, think: bool = False,
             cancelled: Callable[[], bool] = lambda: False
             ) -> Iterator[tuple[str, str]]:
        """Yields ('say', sentence) as each completes and ('tool', name) when
        a tool runs. Stops quietly if cancelled() turns true."""
        stamp = datetime.now().strftime("%a %-d %b %Y, %H:%M")
        this = [{"role": "user", "content": f"[{stamp}] {text}"}]

        for _ in range(C.AGENT_MAX_STEPS):
            msgs = [self.system, *self._history(), *this]
            content, calls = yield from self._stream(msgs, think, cancelled)
            if content is None:                 # cancelled mid-stream
                return
            if not calls:
                this.append({"role": "assistant", "content": content})
                self.turns.append(this)
                self._trim()
                return

            this.append({"role": "assistant", "content": content or None,
                         "tool_calls": [
                             {"id": c["id"], "type": "function",
                              "function": {"name": c["name"],
                                           "arguments": c["args"] or "{}"}}
                             for c in calls]})
            for c in calls:
                if cancelled():
                    return
                yield ("tool", c["name"])
                this.append({"role": "tool", "tool_call_id": c["id"],
                             "content": T.run(c["name"], c["args"])})
            think = False        # thinking once per turn is enough

        yield ("say", "I went round in circles on that one. Try rephrasing it.")

    def _stream(self, msgs, think, cancelled):
        stream = local().chat.completions.create(
            model=C.LOCAL_MODEL, messages=msgs, tools=self.tools, stream=True,
            **sampling(think, C.AGENT_THINK_MAX_TOKENS if think
                       else C.AGENT_MAX_TOKENS))

        # With thinking on, the server either returns reasoning in a separate
        # field (then content is the answer) or inline before </think>. Until
        # we know which, hold content back rather than speak the reasoning.
        holding = think
        pending, buf, full = "", "", ""
        calls: dict[int, dict] = {}
        try:
            for chunk in stream:
                if cancelled():
                    return None, None
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                for tc in d.tool_calls or []:
                    slot = calls.setdefault(tc.index, {
                        "id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
                extra = getattr(d, "model_extra", None) or {}
                if holding and (extra.get("reasoning_content")
                                or extra.get("reasoning")):
                    holding = False                 # reasoning is separate
                    buf, pending = pending, ""
                piece = d.content or ""
                if not piece:
                    continue
                full += piece
                if holding:
                    pending += piece
                    if "</think>" not in pending:
                        continue
                    holding = False
                    buf = pending.split("</think>", 1)[1].lstrip()
                    pending = ""
                else:
                    buf += piece
                buf = THINK_BLOCK.sub("", buf)
                speak, sep, rest = buf.partition("<think>")
                parts = SENTENCE_END.split(speak)
                for s in parts[:-1]:
                    if _clean(s):
                        yield ("say", _clean(s))
                buf = parts[-1] + sep + rest
        finally:
            stream.close()

        if holding:                                 # never saw a boundary
            buf = THINK_BLOCK.sub("", pending).replace("<think>", "")
        tail = _clean(THINK_BLOCK.sub("", buf).split("<think>")[0])
        if tail:
            yield ("say", tail)

        answer = THINK_BLOCK.sub("", full).split("</think>")[-1].strip()
        out = []
        for i in sorted(calls):
            c = calls[i]
            c["id"] = c["id"] or f"call_{i}"
            if c["name"]:
                out.append(c)
        return answer, out
