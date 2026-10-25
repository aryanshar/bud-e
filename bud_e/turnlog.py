"""Per-turn timing, so you tune the slow stage instead of the one that feels
slow. Every voice turn appends a line to ~/.bud-e/turns.jsonl."""

from __future__ import annotations

import json
import math
import statistics
import time
from contextlib import contextmanager
from datetime import datetime

from bud_e.config import Config

STAGES = ["speaker", "asr", "first_audio", "total"]


class Turn:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.times: dict[str, float] = {}
        self.meta: dict[str, object] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = time.perf_counter() - t

    def mark(self, name: str) -> None:
        """Seconds from the end of your speech to this point."""
        self.times.setdefault(name, time.perf_counter() - self.t0)

    def line(self) -> str:
        return "  [" + "  ".join(f"{k} {v:.2f}s" for k, v in self.times.items()
                                 ) + "]"

    def save(self, cfg: Config) -> None:
        self.times["total"] = time.perf_counter() - self.t0
        cfg.ensure_home()
        rec = {"when": datetime.now().isoformat(timespec="seconds"),
               **{k: round(v, 3) for k, v in self.times.items()}, **self.meta}
        with cfg.turn_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def p90(v: list[float]) -> float:
    """Nearest-rank 90th percentile: the value 90% of turns are at or under."""
    return sorted(v)[max(0, math.ceil(len(v) * 0.9) - 1)]


def stats(cfg: Config) -> None:
    if not cfg.turn_log.exists():
        raise SystemExit("no turns logged yet: run `bud-e run` first")
    rows = []
    for line in cfg.turn_log.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    rows = [r for r in rows if "kind" in r]     # skip the old format
    if not rows:
        raise SystemExit("no turns in the current format yet")

    print(f"\n{len(rows)} turns from {cfg.turn_log}\n")
    print(f"  {'kind':<9} {'stage':<12} {'median':>8} {'p90':>8}   n")
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        for stage in STAGES:
            v = [r[stage] for r in sub if stage in r]
            if v:
                print(f"  {kind:<9} {stage:<12} {statistics.median(v):7.2f}s "
                      f"{p90(v):7.2f}s  {len(v):3d}")
    print("\n  first_audio is what a turn feels like: end of your speech to "
          "its first word.\n")
