"""What the model can do. Local, no API keys, each tool small enough to read
in a minute.

Adding one: write a function that takes (cfg, **args) and returns a short
string, and decorate it with @tool. `available` decides whether it is offered
on this machine, so platform-specific tools never reach a model that can't
use them. Results go straight back into the model's context, so keep them
short.

Deliberately absent: a shell tool. Voice dispatch means nobody reads the
command before it runs. File tools are read-only and fenced to read_roots.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from bud_e.config import IS_LINUX, IS_MAC, Config
from bud_e.llm import parse_args

RESULT_CHARS = 4000


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    required: list[str]
    fn: Callable[..., str]
    available: Callable[[Config], bool]

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params,
                           "required": self.required}}}


ALL: list[Tool] = []
STR = {"type": "string"}


def tool(description: str, params: dict | None = None,
         required: tuple[str, ...] = (),
         available: Callable[[Config], bool] = lambda cfg: True):
    def wrap(fn):
        ALL.append(Tool(fn.__name__, description, params or {},
                        list(required), fn, available))
        return fn
    return wrap


class Toolbox:
    """The tools enabled for one config."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tools = {t.name: t for t in ALL
                      if t.name not in cfg.tools.disabled and t.available(cfg)}

    @property
    def names(self) -> list[str]:
        return list(self.tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def run(self, name: str, arguments: str) -> str:
        t = self.tools.get(name)
        if t is None:
            return f"error: no tool named {name}"
        try:
            out = t.fn(self.cfg, **parse_args(arguments))
        except Exception as e:
            return f"error: {type(e).__name__}: {str(e)[:200]}"
        out = str(out)
        if len(out) > RESULT_CHARS:
            out = out[:RESULT_CHARS] + "\n[truncated]"
        return out


# --- memory and notes ---------------------------------------------------------

_STOP = set("a an the i my me is are was to of in on for and or what did do "
            "about you your it that this with at by be have has".split())


@tool("Save a lasting fact about the user (preferences, people, plans).",
      {"fact": STR}, ("fact",))
def remember(cfg: Config, fact: str) -> str:
    cfg.ensure_home()
    with cfg.memory_path.open("a") as f:
        f.write(f"- {date.today().isoformat()} {fact.strip()}\n")
    return "saved"


@tool("Search everything remembered about the user.", {"query": STR},
      ("query",))
def recall(cfg: Config, query: str) -> str:
    if not cfg.memory_path.exists():
        return "memory is empty"
    words = set(re.findall(r"\w+", query.lower())) - _STOP
    lines = [ln.strip() for ln in cfg.memory_path.open() if ln.strip()]
    scored = sorted(((len(words & set(re.findall(r"\w+", ln.lower()))), i, ln)
                     for i, ln in enumerate(lines)), reverse=True)
    hits = [ln for score, _, ln in scored if score > 0][:8]
    return "\n".join(hits) or "nothing relevant in memory"


def memory_snapshot(cfg: Config, chars: int = 3000) -> str:
    """The most recent memories, for the system prompt."""
    if not cfg.memory_path.exists():
        return ""
    return cfg.memory_path.read_text().strip()[-chars:]


@tool("Append a note to the user's notes.", {"text": STR}, ("text",))
def add_note(cfg: Config, text: str) -> str:
    cfg.ensure_home()
    with cfg.notes_path.open("a") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} {text.strip()}\n")
    return "noted"


@tool("Read the user's most recent notes.", {"last": {"type": "integer"}})
def read_notes(cfg: Config, last: int = 10) -> str:
    if not cfg.notes_path.exists():
        return "no notes yet"
    return "\n".join(cfg.notes_path.read_text().splitlines()[-int(last):])


# --- files (read-only, inside read_roots) -------------------------------------

_SKIP = {"node_modules", ".git", "__pycache__", ".venv", "venv", "Library",
         ".Trash"}


def _inside_roots(cfg: Config, p: Path) -> bool:
    p = p.resolve()
    roots = [r.resolve() for r in [*cfg.read_roots, cfg.home]]
    return any(p == r or r in p.parents for r in roots)


@tool("Find files by name in the user's documents folders.", {"name": STR},
      ("name",))
def find_files(cfg: Config, name: str) -> str:
    needle, found, visited = name.lower(), [], 0
    for root in cfg.read_roots:
        if not root.is_dir():
            continue
        for dirpath, dirs, files in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in _SKIP and depth < 4]
            found += [str(Path(dirpath) / f) for f in files
                      if needle in f.lower()]
            visited += 1
            if len(found) >= 15 or visited > 5000:
                break
    return "\n".join(found[:15]) or f"no file matching '{name}'"


@tool("Read a text file, by full path or by file name.", {"path": STR},
      ("path",))
def read_file(cfg: Config, path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute() or not p.exists():
        hits = find_files(cfg, p.name).splitlines()
        if not hits or hits[0].startswith("no file"):
            return f"not found: {path}"
        p = Path(hits[0])
    if not _inside_roots(cfg, p):
        return f"outside the folders I'm allowed to read: {p}"
    if p.is_dir():
        return "\n".join(sorted(x.name for x in p.iterdir())[:50])
    return f"[{p}]\n{p.read_text(errors='replace')}"


# --- weather (Open-Meteo, free, no key) ---------------------------------------

_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "foggy", 51: "drizzle", 53: "drizzle", 55: "drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 71: "snow",
        73: "snow", 75: "heavy snow", 80: "showers", 81: "showers",
        82: "heavy showers", 95: "thunderstorms", 96: "thunderstorms",
        99: "thunderstorms"}


@tool("Current weather and a three day forecast for the user's location.",
      available=lambda cfg: cfg.user.latitude is not None
      and cfg.user.longitude is not None)
def get_weather(cfg: Config) -> str:
    q = urllib.parse.urlencode({
        "latitude": cfg.user.latitude, "longitude": cfg.user.longitude,
        "current": "temperature_2m,apparent_temperature,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max,weather_code",
        "timezone": "auto", "forecast_days": 3})
    with urllib.request.urlopen(
            f"https://api.open-meteo.com/v1/forecast?{q}", timeout=8) as r:
        w = json.load(r)
    cur, day = w["current"], w["daily"]
    lines = [f"{cfg.user.location or 'here'} now: "
             f"{round(cur['temperature_2m'])}C, feels "
             f"{round(cur['apparent_temperature'])}C, "
             f"{_WMO.get(cur['weather_code'], 'mixed')}"]
    for i, d in enumerate(day["time"]):
        lines.append(f"{d}: high {round(day['temperature_2m_max'][i])}C, "
                     f"low {round(day['temperature_2m_min'][i])}C, rain "
                     f"{day['precipitation_probability_max'][i]}%, "
                     f"{_WMO.get(day['weather_code'][i], 'mixed')}")
    return "\n".join(lines)


# --- web (optional, local SearXNG) --------------------------------------------

@tool("Search the web.", {"query": STR}, ("query",),
      available=lambda cfg: bool(cfg.tools.searxng_url))
def web_search(cfg: Config, query: str) -> str:
    q = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"{cfg.tools.searxng_url.rstrip('/')}/search?{q}"
    with urllib.request.urlopen(url, timeout=10) as r:
        results = json.load(r).get("results", [])[:5]
    return "\n\n".join(f"{x.get('title', '')}\n{x.get('content', '')}"
                       for x in results) or "no results"


# --- volume -------------------------------------------------------------------

@tool("Change the speaker volume by a percentage, e.g. 10 or -20.",
      {"change": {"type": "integer"}}, ("change",),
      available=lambda cfg: IS_MAC or (IS_LINUX and bool(shutil.which("pactl"))))
def set_volume(cfg: Config, change: int) -> str:
    change = max(-100, min(100, int(change)))
    if IS_MAC:
        _osa(f"set volume output volume (output volume of "
             f"(get volume settings) + {change})")
    else:
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                        f"{change:+d}%"], check=True, timeout=5)
    return f"volume {change:+d}%"


# --- calendar and reminders (macOS apps) --------------------------------------

def _osa(script: str, timeout: int = 25) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200] or "osascript failed")
    return r.stdout.strip()


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _as_date(var: str, dt: datetime) -> str:
    """AppleScript that builds a date without locale-dependent parsing."""
    return (f"set {var} to current date\n"
            f"set day of {var} to 1\n"
            f"set year of {var} to {dt.year}\n"
            f"set month of {var} to {dt.month}\n"
            f"set day of {var} to {dt.day}\n"
            f"set time of {var} to {dt.hour * 3600 + dt.minute * 60}\n")


def _parse_when(when: str) -> datetime:
    when = when.strip().replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(when)
    except ValueError:      # date only: nine in the morning
        return datetime.combine(date.fromisoformat(when[:10]),
                                datetime.min.time().replace(hour=9))


@tool("List calendar events for one day.",
      {"day": {"type": "string", "description": "YYYY-MM-DD"}}, ("day",),
      available=lambda cfg: IS_MAC)
def get_calendar(cfg: Config, day: str) -> str:
    d = date.fromisoformat(day[:10])
    if shutil.which("icalBuddy"):          # fast, includes recurring events
        r = subprocess.run(
            ["icalBuddy", "-nc", "-nrd", "-b", "", "-ea", "-iep",
             "title,datetime", "-po", "datetime,title", "-tf", "%H:%M",
             "-df", "", f"eventsFrom:{d}", f"to:{d}"],
            capture_output=True, text=True, timeout=20)
        events = r.stdout.strip()
    else:                                  # slower, misses recurring events
        d0 = datetime.combine(d, datetime.min.time())
        events = _osa(_as_date("d0", d0) + """
set d1 to d0 + 1 * days
set out to ""
tell application "Calendar"
  repeat with c in calendars
    repeat with e in (every event of c whose start date >= d0 and start date < d1)
      set out to out & (time string of (start date of e)) & "  " & (summary of e) & linefeed
    end repeat
  end repeat
end tell
return out""", timeout=40)
    return events or f"nothing on {d:%A %-d %B}"


@tool("Create a reminder in Reminders.",
      {"text": STR,
       "when": {"type": "string", "description": "YYYY-MM-DDTHH:MM, optional"}},
      ("text",), available=lambda cfg: IS_MAC)
def set_reminder(cfg: Config, text: str, when: str = "") -> str:
    props, script = f"name:{_q(text)}", ""
    if when:
        dt = _parse_when(when)
        script = _as_date("d", dt)
        props += ", due date:d, remind me date:d"
    _osa(script + 'tell application "Reminders" to make new reminder '
                  f"with properties {{{props}}}")
    return f"reminder set: {text}" + (
        f" at {_parse_when(when):%a %-d %b %H:%M}" if when else "")


@tool("List unfinished reminders due by a day (default today).",
      {"day": {"type": "string", "description": "YYYY-MM-DD"}},
      available=lambda cfg: IS_MAC)
def list_reminders(cfg: Config, day: str = "") -> str:
    d = date.fromisoformat(day[:10]) if day else date.today()
    end = datetime.combine(d + timedelta(days=1), datetime.min.time())
    out = _osa(_as_date("d1", end) + """
set out to ""
tell application "Reminders"
  repeat with r in (every reminder whose completed is false and due date < d1)
    set out to out & (name of r) & linefeed
  end repeat
end tell
return out""")
    return out or "no reminders due"
