"""Compare everything in results/ as one table."""

import json

from bud_e import config as C

COLS = ["model", "think", "reply med", "reply p90", "tok/s",
        "behaviour", "malformed"]


def load() -> list[dict]:
    out = []
    for p in sorted(C.RESULTS.glob("*.json")):
        try:
            out.append(json.loads(p.read_text())["summary"])
        except (json.JSONDecodeError, KeyError):
            print(f"  skipping unreadable {p.name}")
    return out


def report() -> None:
    rows = load()
    if not rows:
        raise SystemExit("nothing in results/ — run `bud-bench run` first")

    rows.sort(key=lambda s: (-s["tool_correct"] / s["n"], s["ttft_median"]))
    table = [[
        s["model"],
        "on" if s.get("thinking") else "off",
        f"{s['ttft_median']:.2f}s",
        f"{s['ttft_p90']:.2f}s",
        f"{s['tps_mean']:.1f}",
        f"{s['tool_correct']}/{s['n']} ({s['tool_correct']/s['n']:.0%})",
        str(s["malformed"]),
    ] for s in rows]

    widths = [max(len(str(r[i])) for r in [COLS, *table])
              for i in range(len(COLS))]
    line = "  ".join(c.ljust(w) for c, w in zip(COLS, widths))
    print(f"\n{line}")
    print("-" * len(line))
    for r in table:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(f"\nbest by behaviour then latency: {rows[0]['model']}")
    print(f"set it with: export BUD_E_MODEL={rows[0]['model']}\n")
