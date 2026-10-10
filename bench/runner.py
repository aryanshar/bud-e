"""Measure the three numbers that decide whether a model can be Bud-E's brain:
time to first token, sustained throughput, and tool-call reliability."""

import json
import statistics as stats
import time
from datetime import datetime, timezone

from bud_e import config as C
from bud_e.llm import local, sampling
from bench.cases import CASES, SYSTEM, TOOLS


def _reasoning(delta) -> str:
    """Thinking models put their scratchpad outside `content`."""
    for key in ("reasoning_content", "reasoning"):
        v = getattr(delta, key, None) or (delta.model_extra or {}).get(key)
        if v:
            return v
    return ""


def one_turn(client, model: str, prompt: str, expected, think: bool = False):
    # Qwen's soft switch is parsed from the latest USER message, not the
    # system prompt. Putting it in the system block does nothing.
    from datetime import datetime as _dt
    user = f"[{_dt.now():%a %-d %b %Y, %H:%M}] {prompt}"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    t0 = time.perf_counter()
    ttft = None          # first token of any kind
    ttfc = None          # first spoken token, which is what the user waits for
    n_tok = 0
    n_think = 0
    total_tok = None
    text = ""
    calls: dict[int, dict] = {}

    kwargs = dict(model=model, messages=msgs, tools=TOOLS, stream=True,
                  stream_options={"include_usage": True},
                  **sampling(think, 4096 if think else 1200))

    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("extra_body", None)      # server rejects the knob; /no_think only
        stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        if getattr(chunk, "usage", None):
            total_tok = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        think_delta = _reasoning(d)
        if think_delta:
            n_think += 1
        if ttft is None and (d.content or d.tool_calls or think_delta):
            ttft = time.perf_counter() - t0
        if ttfc is None and (d.content or d.tool_calls):
            ttfc = time.perf_counter() - t0
        if d.content:
            n_tok += 1
            text += d.content
        for tc in (d.tool_calls or []):
            slot = calls.setdefault(tc.index, {"name": "", "args": ""})
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["args"] += tc.function.arguments

    dt = time.perf_counter() - t0
    names = [c["name"] for c in calls.values()]

    malformed = False
    for c in calls.values():
        try:
            json.loads(c["args"] or "{}")
        except json.JSONDecodeError:
            malformed = True

    if expected is None:
        correct = not names and bool(text.strip())
    else:
        ok = expected if isinstance(expected, tuple) else (expected,)
        correct = any(e in names for e in ok)

    gen = total_tok if total_tok else (n_tok + n_think)
    return {"prompt": prompt, "expected": expected, "called": names,
            "ttft": ttft or dt, "ttfc": ttfc or dt, "dt": dt,
            "tokens": n_tok, "thinking": n_think, "generated": gen,
            "tps": gen / dt if dt else 0.0,
            "correct": correct, "malformed": malformed,
            "text": text[:200]}


def run(model: str, runs: int = 3, base: str | None = None,
        save: bool = True, think: bool = False) -> dict:
    C.ensure_dirs()
    client = local(base)
    rows = []

    print(f"\n{model}   {runs} runs x {len(CASES)} cases\n")
    for _ in range(runs):
        for prompt, expected in CASES:
            try:
                res = one_turn(client, model, prompt, expected, think=think)
            except Exception as e:
                print(f"  ERR  {prompt[:44]!r}  {e}")
                continue
            rows.append(res)
            ok = res["correct"] and not res["malformed"]
            think_note = f" +{res['thinking']}thk" if res["thinking"] else ""
            print(f"  {'ok ' if ok else 'BAD'} "
                  f"reply {res['ttfc']:5.2f}s  {res['tps']:5.1f} tok/s"
                  f"{think_note}  "
                  f"want {res['expected'] or '<answer>'}  "
                  f"got {','.join(res['called']) or '<text>'}")

    if not rows:
        raise SystemExit("no successful turns — check the endpoint")

    ttfts = sorted(r["ttfc"] for r in rows)
    summary = {
        "model": model,
        "thinking": bool(think),
        "think_tokens": sum(r["thinking"] for r in rows),
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
        "n": len(rows),
        "ttft_median": stats.median(ttfts),
        "ttft_p90": ttfts[max(0, int(len(ttfts) * 0.9) - 1)],
        "tps_mean": stats.mean(r["tps"] for r in rows),
        "tool_correct": sum(r["correct"] for r in rows),
        "malformed": sum(r["malformed"] for r in rows),
    }
    print_summary(summary)

    if save:
        slug = model.replace("/", "_").replace(":", "-")
        path = C.RESULTS / f"{slug}.json"
        path.write_text(json.dumps({"summary": summary, "rows": rows},
                                   indent=2))
        print(f"  saved {path.relative_to(C.REPO)}\n")
    return summary


def print_summary(s: dict) -> None:
    acc = s["tool_correct"] / s["n"]
    verdict = "USABLE" if (acc >= 0.85 and s["ttft_median"] < 1.0
                           and s["tps_mean"] > 20) else "MARGINAL"
    print("\n  ---")
    print(f"  reply       median {s['ttft_median']:.2f}s   p90 {s['ttft_p90']:.2f}s")
    print(f"  throughput  mean {s['tps_mean']:.1f} tok/s")
    if s.get("think_tokens"):
        print(f"  thinking    {s['think_tokens']} tokens burned before speaking")
    print(f"  behaviour   {s['tool_correct']}/{s['n']} ({acc:.0%})")
    print(f"  malformed   {s['malformed']}/{s['n']}")
    print(f"  verdict     {verdict}")
    print("  targets: reply < 1.50s, tps > 20, behaviour >= 85%, malformed 0")
