"""Does this model deserve the job? Three numbers decide it:

  behaviour  picks the right tool, or answers plainly when no tool fits
  latency    time until it starts talking (or has a tool call ready)
  speed      decode tokens per second (speech needs about 5)

Runs the agent's real system prompt and tool schemas, so a score here means
something for the voice path. Results go to ~/.bud-e/bench/.
"""

from __future__ import annotations

import json
import statistics
import time

from bud_e.agent import Agent, stamp
from bud_e.config import Config
from bud_e.llm import LLM, parse_args
from bud_e.turnlog import p90

# (what you'd say, the tool it should call; None = answer without tools)
CASES = [
    ("What's on my calendar tomorrow?", "get_calendar"),
    ("Remind me to ring the plumber on Tuesday at four.", "set_reminder"),
    ("What did I tell you about the flat last week?", "recall"),
    ("Read me what's in notes dot txt.", ("read_file", "find_files")),
    ("Will it rain this afternoon?", "get_weather"),
    ("By the way, I take my coffee black, no sugar.", "remember"),
    ("Turn it up a bit.", "set_volume"),
    ("Roughly how far away is the moon?", None),
    ("Explain what a mixture of experts model is in two sentences.", None),
    ("Should I learn Rust or Go first?", None),
]


def _one(agent: Agent, prompt: str) -> dict:
    msgs = [agent.system, {"role": "user", "content": f"[{stamp()}] {prompt}"}]
    t0, first, text, calls = time.perf_counter(), None, "", []
    for kind, value in agent.llm.stream(msgs, agent.tools.schemas() or None):
        if kind == "text":
            first = first or time.perf_counter()
            text += value
        else:
            calls = value
    end = time.perf_counter()
    malformed = 0
    for c in calls:
        try:
            parse_args(c.arguments)
        except ValueError:
            malformed += 1
    return {"latency": (first or end) - t0, "text": text.strip()[:200],
            "called": [c.name for c in calls], "malformed": malformed}


def _decode_speed(llm: LLM) -> float:
    msgs = [{"role": "user", "content":
             "Write about two hundred words on the history of tea."}]
    try:
        stream = llm.client.chat.completions.create(
            model=llm.model, messages=msgs, max_tokens=300, stream=True,
            stream_options={"include_usage": True},
            extra_body=llm._extra or None)
    except Exception:
        return 0.0
    first, n = None, 0
    for ch in stream:
        if ch.choices and ch.choices[0].delta.content:
            first = first or time.perf_counter()
        if getattr(ch, "usage", None):
            n = ch.usage.completion_tokens
    end = time.perf_counter()
    return (n - 1) / (end - first) if first and n > 1 and end > first else 0.0


def run(cfg: Config, runs: int = 2) -> dict:
    agent = Agent(cfg)
    names = set(agent.tools.names)
    cases = [(p, e) for p, e in CASES
             if e is None or set(e if isinstance(e, tuple) else (e,)) & names]
    print(f"\n{agent.llm.model}: {runs} runs x {len(cases)} cases "
          f"({len(CASES) - len(cases)} skipped, tools not on this machine)\n")
    agent.warm()

    rows = []
    for _ in range(runs):
        for prompt, expected in cases:
            r = _one(agent, prompt)
            ok_tools = expected if isinstance(expected, tuple) else (expected,)
            r["ok"] = (not r["called"] and bool(r["text"])) if expected is None \
                else any(t in r["called"] for t in ok_tools)
            r["ok"] = r["ok"] and not r["malformed"]
            rows.append(r)
            got = ",".join(r["called"]) or "<answer>"
            print(f"  {'ok ' if r['ok'] else 'BAD'} {r['latency']:5.2f}s  "
                  f"want {expected or '<answer>'}, got {got}")

    lat = [r["latency"] for r in rows]
    summary = {
        "model": agent.llm.model, "base_url": cfg.llm.base_url,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n": len(rows), "behaviour": sum(r["ok"] for r in rows) / len(rows),
        "malformed": sum(r["malformed"] for r in rows),
        "latency_median": statistics.median(lat),
        "latency_p90": p90(lat),
        "decode_tps": _decode_speed(agent.llm),
    }
    good = (summary["behaviour"] >= 0.85 and summary["malformed"] == 0
            and summary["latency_median"] < 1.5)
    print(f"""
  behaviour  {summary['behaviour']:.0%}   (want 85%+)
  malformed  {summary['malformed']}      (want 0)
  latency    median {summary['latency_median']:.2f}s, p90 {summary['latency_p90']:.2f}s   (want under 1.5s)
  decode     {summary['decode_tps']:.0f} tok/s   (speech needs ~5, 20+ feels snappy)
  verdict    {'GOOD' if good else 'MARGINAL'}
""")
    out = cfg.home / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (agent.llm.model.replace("/", "_").replace(":", "-") + ".json")
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"  saved {path}\n")
    return summary


def report(cfg: Config) -> None:
    rows = []
    for p in sorted((cfg.home / "bench").glob("*.json")):
        try:
            rows.append(json.loads(p.read_text())["summary"])
        except (json.JSONDecodeError, KeyError):
            pass
    if not rows:
        raise SystemExit("no results yet: run `bud-e bench` first")
    rows.sort(key=lambda s: (-s["behaviour"], s["latency_median"]))
    print(f"\n  {'model':<44} {'behaviour':>9} {'latency':>8} {'tok/s':>6}")
    for s in rows:
        print(f"  {s['model'][:44]:<44} {s['behaviour']:>9.0%} "
              f"{s['latency_median']:>7.2f}s {s['decode_tps']:>6.0f}")
    print()
