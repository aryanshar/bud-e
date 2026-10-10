"""Phrases you say often should not cost a model round trip.

A regex hit returns in milliseconds. A model turn costs a second or more.
Anything you say daily with a predictable shape and no judgement belongs here.
Anything that needs parsing (reminders, briefs) goes to the agent instead.

Return a string to speak, or "" to say nothing.
"""

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from bud_e import config as C
from bud_e.tools import weather_summary


@dataclass
class Trigger:
    pattern: re.Pattern
    action: Callable[[re.Match], str]


def _time(m) -> str:
    return f"It's {datetime.now().strftime('%-I:%M')}, {C.ADDRESS}."


def _date(m) -> str:
    return datetime.now().strftime("It's %A the %-d of %B.")


def _volume(m) -> str:
    up = bool(re.search(r"\b(up|louder|increase)\b", m.group(0), re.I))
    delta = 10 if up else -10
    subprocess.run(["osascript", "-e",
                    f"set volume output volume (output volume of "
                    f"(get volume settings) + {delta})"])
    return ""


def _sleep(m) -> str:
    subprocess.run(["pmset", "displaysleepnow"])
    return ""


TRIGGERS: list[Trigger] = [
    # --- silence and control -------------------------------------------
    Trigger(re.compile(r"^\s*(stop|quiet|shut up|never ?mind|forget it)\b", re.I),
            lambda m: ""),
    Trigger(re.compile(r"^\s*(thanks|thank you|cheers|nice one)( bud-?e| buddy)?[.!]?\s*$", re.I),
            lambda m: ""),
    Trigger(re.compile(r"\b(volume|louder|quieter) ?(up|down)?\b|"
                       r"\b(turn it (up|down))\b", re.I), _volume),
    Trigger(re.compile(r"\b(go to sleep|lock the screen|screen off)\b", re.I),
            _sleep),

    # --- instant answers ------------------------------------------------
    Trigger(re.compile(r"\bwhat('s| is) the time\b|\bwhat time is it\b", re.I),
            _time),
    Trigger(re.compile(r"\bwhat('s| is) (the date|today)\b|"
                       r"\bwhat day is it\b", re.I), _date),

    Trigger(re.compile(r"^\s*(what'?s|how'?s) the weather( like)?( today| now)?\??\s*$",
                       re.I), lambda m: weather_summary()),
]


def match(text: str):
    for t in TRIGGERS:
        m = t.pattern.search(text)
        if m:
            return t, m
    return None, None
