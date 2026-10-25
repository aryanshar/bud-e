"""The model server. Anything that speaks the OpenAI chat API works:
oMLX, LM Studio, Ollama, mlx_lm.server, llama.cpp's llama-server, vLLM.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

from openai import BadRequestError, OpenAI

from bud_e.config import Config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON text, exactly as the model wrote it

    def as_message(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name,
                             "arguments": self.arguments or "{}"}}


class LLM:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.llm
        self.client = OpenAI(base_url=self.cfg.base_url,
                             api_key=self.cfg.api_key,
                             timeout=self.cfg.timeout)
        self._extra = dict(self.cfg.extra_body)
        self._model = self.cfg.model

    @property
    def model(self) -> str:
        if not self._model:
            models = self.models()
            if not models:
                raise RuntimeError(f"{self.cfg.base_url} serves no models")
            self._model = models[0]
        return self._model

    def models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def _create(self, **kw):
        kw = dict(model=self.model, temperature=self.cfg.temperature,
                  top_p=self.cfg.top_p, **kw)
        kw.setdefault("max_tokens", self.cfg.max_tokens)
        try:
            return self.client.chat.completions.create(
                extra_body=self._extra or None, **kw)
        except BadRequestError:
            if not self._extra:
                raise
            # Some servers reject fields they don't know. Drop them for good.
            print("  [llm] server rejected extra_body, retrying without it")
            self._extra = {}
            return self.client.chat.completions.create(**kw)

    def stream(self, messages: list[dict], tools: list[dict] | None = None,
               max_tokens: int | None = None,
               ) -> Iterator[tuple[str, object]]:
        """Yields ('text', str) as content arrives, then exactly one
        ('calls', list[ToolCall]) at the end (empty if none).

        Reasoning sent in a separate field is ignored. Inline <think> tags
        are the caller's job (see text.ThinkFilter)."""
        kw = {"messages": messages, "stream": True}
        if tools:
            kw["tools"] = tools
        if max_tokens:
            kw["max_tokens"] = max_tokens
        stream = self._create(**kw)

        slots: dict[int, dict] = {}
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                for tc in d.tool_calls or []:
                    s = slots.setdefault(tc.index or 0,
                                         {"id": "", "name": "", "args": ""})
                    s["id"] = tc.id or s["id"]
                    if tc.function and tc.function.name:
                        s["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        s["args"] += tc.function.arguments
                if d.content:
                    yield "text", d.content
        finally:
            stream.close()

        calls = [ToolCall(s["id"] or f"call_{i}", s["name"], s["args"])
                 for i, s in sorted(slots.items()) if s["name"]]
        yield "calls", calls

    def complete(self, messages: list[dict], **kw) -> str:
        """One non-streamed reply. For checks, not for the voice path."""
        r = self._create(messages=messages, **kw)
        return r.choices[0].message.content or ""


def parse_args(raw: str) -> dict:
    """Tool arguments as a dict. Raises ValueError with a message the model
    can act on if it wrote broken JSON."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(f"arguments were not valid JSON ({e.msg})") from None
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    return args
