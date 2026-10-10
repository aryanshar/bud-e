"""Bud-E's tools. Local, no API keys, each one small enough to read in a minute.

Every tool takes plain JSON arguments and returns a short string, because the
result goes straight back into the 27B's context and costs prefill.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from bud_e import config as C

# Set by the voice loop so a finished background job can be announced.
on_delegate_done: Callable[[str], None] | None = None


# --- helpers -------------------------------------------------------------

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
    except ValueError:
        return datetime.combine(date.fromisoformat(when[:10]),
                                datetime.min.time().replace(hour=9))


def _allowed(p: Path) -> bool:
    p = p.resolve()
    roots = [*C.READ_ROOTS, C.HOME]
    return any(p == r.resolve() or r.resolve() in p.parents for r in roots)


# --- memory and notes ------------------------------------------------------

STOP = set("a an the i my me is are was to of in on for and or what did do "
           "about you your it that this with at by be have has".split())


def remember(fact: str) -> str:
    C.ensure_dirs()
    with C.MEMORY_PATH.open("a") as f:
        f.write(f"- {date.today().isoformat()} {fact.strip()}\n")
    return "saved"


def recall(query: str) -> str:
    if not C.MEMORY_PATH.exists():
        return "memory is empty"
    words = set(re.findall(r"\w+", query.lower())) - STOP
    lines = [ln.strip() for ln in C.MEMORY_PATH.open() if ln.strip()]
    scored = sorted(((len(words & set(re.findall(r"\w+", ln.lower()))), i, ln)
                     for i, ln in enumerate(lines)), reverse=True)
    hits = [ln for s, _, ln in scored if s > 0][:8]
    return "\n".join(hits) if hits else "nothing relevant in memory"


def add_note(text: str) -> str:
    C.ensure_dirs()
    with C.NOTES_PATH.open("a") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} {text.strip()}\n")
    return f"added to {C.NOTES_PATH}"


def read_notes(last: int = 10) -> str:
    if not C.NOTES_PATH.exists():
        return "no notes yet"
    lines = C.NOTES_PATH.read_text().strip().splitlines()
    return "\n".join(lines[-int(last):])


# --- files (read-only, inside READ_ROOTS) ----------------------------------

SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv",
             "Library", ".Trash"}


def find_files(name: str) -> str:
    needle, out, seen = name.lower(), [], 0
    for root in C.READ_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirs, files in os.walk(root):
            depth = Path(dirpath).relative_to(root).parts
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in SKIP_DIRS and len(depth) < 4]
            seen += 1
            out += [str(Path(dirpath) / f) for f in files if needle in f.lower()]
            if len(out) >= 15 or seen > 5000:
                break
    return "\n".join(out[:15]) or f"no file matching '{name}'"


def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute() or not p.exists():
        found = find_files(p.name).splitlines()
        if not found or found[0].startswith("no file"):
            return f"not found: {path}"
        p = Path(found[0])
    if not _allowed(p):
        return f"outside the allowed folders: {p}"
    if p.is_dir():
        return "\n".join(sorted(x.name for x in p.iterdir())[:50])
    text = p.read_text(errors="replace")
    return f"[{p}]\n{text}"


# --- calendar and reminders (macOS apps via AppleScript) -------------------

def get_calendar(day: str) -> str:
    d = date.fromisoformat(day[:10])
    if shutil.which("icalBuddy"):          # fast, and includes recurring events
        r = subprocess.run(
            ["icalBuddy", "-nc", "-nrd", "-b", "", "-ea", "-iep",
             "title,datetime", "-po", "datetime,title", "-tf", "%H:%M",
             "-df", "", f"eventsFrom:{d}", f"to:{d}"],
            capture_output=True, text=True, timeout=20)
        events = r.stdout.strip()
    else:                                  # slower; misses recurring events
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


def set_reminder(text: str, when: str = "") -> str:
    props = f"name:{_q(text)}"
    script = ""
    if when:
        dt = _parse_when(when)
        script = _as_date("d", dt)
        props += ", due date:d, remind me date:d"
    _osa(script + f'tell application "Reminders" to make new reminder '
                  f"with properties {{{props}}}")
    return f"reminder set: {text}" + (f" at {_parse_when(when):%a %-d %b %H:%M}"
                                      if when else "")


def list_reminders(day: str = "") -> str:
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


# --- weather (Open-Meteo, free, no key) ------------------------------------

WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
       45: "foggy", 48: "foggy", 51: "drizzle", 53: "drizzle", 55: "drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 80: "showers",
       81: "showers", 82: "heavy showers", 95: "thunderstorms",
       96: "thunderstorms", 99: "thunderstorms"}


def _weather_raw() -> dict:
    q = urllib.parse.urlencode({
        "latitude": C.WEATHER_LAT, "longitude": C.WEATHER_LON,
        "current": "temperature_2m,apparent_temperature,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max,weather_code",
        "timezone": "auto", "forecast_days": 3})
    with urllib.request.urlopen(
            f"https://api.open-meteo.com/v1/forecast?{q}", timeout=8) as r:
        return json.load(r)


def get_weather(day: str = "") -> str:
    w = _weather_raw()
    cur, daily = w["current"], w["daily"]
    lines = [f"{C.WEATHER_PLACE} now: {round(cur['temperature_2m'])}C, feels "
             f"{round(cur['apparent_temperature'])}C, "
             f"{WMO.get(cur['weather_code'], 'mixed')}"]
    for i, dd in enumerate(daily["time"]):
        lines.append(f"{dd}: high {round(daily['temperature_2m_max'][i])}C, "
                     f"low {round(daily['temperature_2m_min'][i])}C, rain "
                     f"{daily['precipitation_probability_max'][i]}%, "
                     f"{WMO.get(daily['weather_code'][i], 'mixed')}")
    return "\n".join(lines)


def weather_summary() -> str:
    """Spoken one-liner for the trigger tier. No model involved."""
    try:
        w = _weather_raw()
    except Exception:
        return "I can't reach the weather service right now."
    cur, daily = w["current"], w["daily"]
    rain = daily["precipitation_probability_max"][0]
    return (f"It's {round(cur['temperature_2m'])} degrees and "
            f"{WMO.get(cur['weather_code'], 'mixed')}. High of "
            f"{round(daily['temperature_2m_max'][0])}, {rain} percent "
            f"chance of rain.")


# --- web (optional, local SearXNG) -----------------------------------------

def web_search(query: str) -> str:
    q = urllib.parse.urlencode({"q": query, "format": "json"})
    with urllib.request.urlopen(f"{C.SEARXNG_URL}/search?{q}", timeout=10) as r:
        res = json.load(r).get("results", [])[:5]
    return "\n\n".join(f"{x.get('title', '')}\n{x.get('content', '')}"
                       for x in res) or "no results"


# --- background delegate (Hermes) ------------------------------------------

def delegate(task: str) -> str:
    """Long multi-step jobs go to Hermes in the background. The voice loop
    stays free; the result is announced and written to the inbox."""
    if not shutil.which("hermes"):
        return "hermes is not installed, so I can't run background jobs"

    def work() -> None:
        cmd = ["hermes", "chat", "--oneshot", "-Q", "-c", C.HERMES_SESSION,
               "--create-if-missing", "-t", C.HERMES_TOOLSETS,
               "-s", C.VOICE_SKILL, "-q", task]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=C.HERMES_TIMEOUT_S)
            result = r.stdout.strip() if r.returncode == 0 else \
                f"failed: {(r.stderr or r.stdout).strip()[:200]}"
        except subprocess.TimeoutExpired:
            result = "timed out"
        C.ensure_dirs()
        with C.INBOX_PATH.open("a") as f:
            f.write(f"## {datetime.now():%Y-%m-%d %H:%M} {task}\n{result}\n\n")
        if on_delegate_done:
            on_delegate_done(f"The background job on {task[:60]} is done. "
                             f"{result[:400]}")

    threading.Thread(target=work, daemon=True).start()
    return "started in the background"


# --- registry --------------------------------------------------------------

def _fn(name, desc, props=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {},
                       "required": list(required)}}}


S = {"type": "string"}
REGISTRY: dict[str, tuple[Callable[..., str], dict]] = {
    "remember": (remember, _fn(
        "remember", "Save a durable fact about the user to long-term memory.",
        {"fact": S}, ["fact"])),
    "recall": (recall, _fn(
        "recall", "Search long-term memory of the user and past conversations.",
        {"query": S}, ["query"])),
    "add_note": (add_note, _fn(
        "add_note", "Append a note to the user's notes file.",
        {"text": S}, ["text"])),
    "read_notes": (read_notes, _fn(
        "read_notes", "Read the most recent notes.",
        {"last": {"type": "integer"}})),
    "find_files": (find_files, _fn(
        "find_files", "Find files by name in Documents, Desktop and Downloads.",
        {"name": S}, ["name"])),
    "read_file": (read_file, _fn(
        "read_file", "Read a text file by path or by file name.",
        {"path": S}, ["path"])),
    "get_calendar": (get_calendar, _fn(
        "get_calendar", "List calendar events for one day.",
        {"day": {"type": "string", "description": "YYYY-MM-DD"}}, ["day"])),
    "set_reminder": (set_reminder, _fn(
        "set_reminder", "Create a reminder in Reminders.app.",
        {"text": S, "when": {"type": "string",
                             "description": "YYYY-MM-DDTHH:MM, optional"}},
        ["text"])),
    "list_reminders": (list_reminders, _fn(
        "list_reminders", "List incomplete reminders due by a day (default today).",
        {"day": {"type": "string", "description": "YYYY-MM-DD"}})),
    "get_weather": (get_weather, _fn(
        "get_weather", "Current weather and a three day forecast.")),
}
if C.SEARXNG_URL:
    REGISTRY["web_search"] = (web_search, _fn(
        "web_search", "Search the web.", {"query": S}, ["query"]))
if C.HERMES_DELEGATE and shutil.which("hermes"):
    REGISTRY["delegate"] = (delegate, _fn(
        "delegate", "Hand a long multi-step job to a background agent. Use only "
        "when the user is happy to hear back later.", {"task": S}, ["task"]))


def schemas() -> list[dict]:
    return [spec for _, spec in REGISTRY.values()]


def run(name: str, arguments: str) -> str:
    if name not in REGISTRY:
        return f"error: no tool named {name}"
    try:
        args = json.loads(arguments or "{}")
        out = REGISTRY[name][0](**args)
    except Exception as e:
        return f"error: {type(e).__name__}: {str(e)[:200]}"
    return out[:C.TOOL_RESULT_CHARS]
