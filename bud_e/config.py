"""All settings, loaded once from ~/.bud-e/config.toml.

Every key is optional. Defaults live in the dataclasses below, so this file is
also the reference for what can be set. A few environment variables override
the file, which is handy for trying another model without editing anything:

    BUD_E_CONFIG    path to the config file
    BUD_E_HOME      where state lives (default ~/.bud-e)
    BUD_E_BASE_URL  model server, e.g. http://localhost:11434/v1
    BUD_E_MODEL     model id as the server lists it
    BUD_E_API_KEY   key for the model server, if it wants one
"""

from __future__ import annotations

import json
import os
import platform
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from urllib.parse import urlparse

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


@dataclass
class LLM:
    base_url: str = "http://localhost:11434/v1"
    model: str = ""            # empty: use the first model the server lists
    api_key: str = ""
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 400
    timeout: float = 120.0
    # oMLX caches prompts of hybrid models (Qwen3.5+, Gated DeltaNet) in
    # 2048-token blocks and skips anything shorter, so a short prompt is
    # re-read in full every turn. Set to ~2100 there to pad past one block.
    # Other servers don't need it: leave 0.
    min_prompt_tokens: int = 0
    # Sent with every request. The default turns off Qwen-style thinking,
    # which on a voice path is just silence. Servers that don't know the
    # field ignore it; if yours rejects it, set extra_body = {}.
    extra_body: dict = field(default_factory=lambda: {
        "chat_template_kwargs": {"enable_thinking": False}})


@dataclass
class Assistant:
    name: str = "Bud-E"
    wake_words: list[str] = field(
        default_factory=lambda: ["bud-e", "buddy", "jarvis"])
    followup_seconds: float = 30.0     # no wake word needed inside this
    history_turns: int = 12
    max_tool_steps: int = 5
    persona: str = ""                  # extra style notes for the prompt


@dataclass
class User:
    name: str = ""                     # what it calls you
    address: str = ""                  # e.g. "sir", used sparingly
    location: str = ""                 # place name for weather
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class Voice:
    asr: str = "auto"                  # auto | mlx-whisper | faster-whisper
    whisper_model: str = "large-v3-turbo"
    language: str = "en"
    tts: str = "auto"                  # auto | kokoro | say | none
    kokoro_voice: str = "bm_george"    # a/b prefix = American/British
    kokoro_speed: float = 1.05
    say_voice: str = "Daniel"          # macOS `say` fallback
    vad_threshold: float = 0.5
    silence_ms: int = 450              # pause that ends your turn
    min_speech_s: float = 0.6
    max_speech_s: float = 20.0
    barge_rms: float = 0.03            # loudness that counts as you talking
    barge_rms_speaking: float = 0.08   # same, while it is talking (echo)
    echo_tail_ms: int = 300            # mic ignored this long after it speaks
    speaker_check_min_s: float = 1.0   # shorter clips skip the voice check
    speaker_check: str = "auto"        # auto (on once enrolled) | on | off
    input_device: str = ""             # name or index; empty = default
    output_device: str = ""


@dataclass
class Tools:
    read_roots: list[str] = field(
        default_factory=lambda: ["~/Documents", "~/Desktop", "~/Downloads"])
    searxng_url: str = ""              # local SearXNG enables web_search
    disabled: list[str] = field(default_factory=list)


@dataclass
class Config:
    llm: LLM = field(default_factory=LLM)
    assistant: Assistant = field(default_factory=Assistant)
    user: User = field(default_factory=User)
    voice: Voice = field(default_factory=Voice)
    tools: Tools = field(default_factory=Tools)
    home: Path = Path.home() / ".bud-e"
    source: Path | None = None         # the file it was loaded from

    # --- state files -----------------------------------------------------
    @property
    def memory_path(self) -> Path:
        return self.home / "memory.md"

    @property
    def notes_path(self) -> Path:
        return self.home / "notes.md"

    @property
    def vocab_path(self) -> Path:
        return self.home / "vocab.txt"

    @property
    def voiceprint_path(self) -> Path:
        return self.home / "voiceprint.npy"

    @property
    def threshold_path(self) -> Path:
        return self.home / "threshold.txt"

    @property
    def turn_log(self) -> Path:
        return self.home / "turns.jsonl"

    @property
    def read_roots(self) -> list[Path]:
        return [Path(p).expanduser() for p in self.tools.read_roots]

    def ensure_home(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)


def _merge(obj, data: dict, where: str) -> None:
    """Copy known keys from a TOML table onto a dataclass, recursively."""
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            print(f"  [config] unknown key {where}{key}, ignored")
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def _omlx_key(base_url: str) -> str:
    """oMLX keeps its API key in its own settings file. Reading it saves
    copying a secret around when the server is oMLX on this machine."""
    p = Path.home() / ".omlx" / "settings.json"
    if not p.exists():
        return ""
    try:
        s = json.loads(p.read_text())
        if urlparse(base_url).port == s.get("server", {}).get("port"):
            return s.get("auth", {}).get("api_key", "")
    except Exception:
        pass
    return ""


def config_path() -> Path:
    if os.environ.get("BUD_E_CONFIG"):
        return Path(os.environ["BUD_E_CONFIG"]).expanduser()
    home = Path(os.environ.get("BUD_E_HOME", Path.home() / ".bud-e"))
    return home.expanduser() / "config.toml"


def load(path: Path | None = None) -> Config:
    cfg = Config()
    if os.environ.get("BUD_E_HOME"):
        cfg.home = Path(os.environ["BUD_E_HOME"]).expanduser()

    path = path or config_path()
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
        if "home" in data:
            data["home"] = Path(data["home"]).expanduser()
        _merge(cfg, data, "")
        cfg.source = path

    env = {"BUD_E_BASE_URL": "base_url", "BUD_E_MODEL": "model",
           "BUD_E_API_KEY": "api_key"}
    for var, key in env.items():
        if os.environ.get(var):
            setattr(cfg.llm, key, os.environ[var])

    cfg.llm.base_url = cfg.llm.base_url.rstrip("/")
    if not cfg.llm.api_key:
        cfg.llm.api_key = _omlx_key(cfg.llm.base_url) or "local"
    return cfg
