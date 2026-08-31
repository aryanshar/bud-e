"""Per-stage timing. Answers which step to optimise instead of guessing."""

import json
import statistics as stats
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

from bud_e import config as C

STAGES = ["embed", "asr", "llm", "tts_start"]


class Turn:
    """One utterance, timed stage by stage."""

    def __init__(self):
        self.t: dict[str, float] = {}
        self.meta: dict[str, object] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            self.t[name] = perf_counter() - t0

    @property
    def total(self) -> float:
        return sum(self.t.values())

    def line(self) -> str:
        parts = " ".join(f"{k} {self.t[k]:.2f}" for k in STAGES if k in self.t)
        return f"  [{parts} | total {self.total:.2f}s]"

    def save(self) -> None:
        C.ensure_dirs()
        rec = {"when": datetime.now().isoformat(timespec="seconds"),
               **{k: round(v, 3) for k, v in self.t.items()},
               "total": round(self.total, 3), **self.meta}
        with C.TURN_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def summarise() -> None:
    if not C.TURN_LOG.exists():
        raise SystemExit(f"no turns logged yet — run `bud-e run` first")

    rows = []
    for line in C.TURN_LOG.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise SystemExit("turn log is empty")

    print(f"\n{len(rows)} turns from {C.TURN_LOG}\n")
    print(f"  {'stage':<12} {'median':>8} {'p90':>8} {'share':>7}")
    print("  " + "-" * 38)

    totals = [r["total"] for r in rows]
    for name in STAGES:
        vals = sorted(r[name] for r in rows if name in r)
        if not vals:
            continue
        med = stats.median(vals)
        p90 = vals[max(0, int(len(vals) * 0.9) - 1)]
        share = sum(vals) / sum(totals) if sum(totals) else 0
        print(f"  {name:<12} {med:7.2f}s {p90:7.2f}s {share:6.0%}")

    print("  " + "-" * 38)
    print(f"  {'total':<12} {stats.median(totals):7.2f}s "
          f"{sorted(totals)[max(0, int(len(totals) * 0.9) - 1)]:7.2f}s")
    print("\n  The largest share is where optimisation pays. Anything under")
    print("  15% is not worth touching however slow it feels.\n")
