import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="bud-e")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enrol", help="record clips and build your voiceprint")
    sub.add_parser("calibrate", help="score yourself vs others, set threshold")
    sub.add_parser("run", help="start the listener")
    sub.add_parser("doctor", help="check the endpoint, models and tools")
    sub.add_parser("profile", help="per-stage timing across logged turns")
    sub.add_parser("latency", help="TTFT cold vs warm, decode speed, fast tier verdict")
    c = sub.add_parser("chat", help="talk to the agent tier in text, no audio")
    c.add_argument("--think", action="store_true")
    a = p.parse_args()

    if a.cmd == "enrol":
        from bud_e.speaker import enrol
        enrol()
    elif a.cmd == "calibrate":
        from bud_e.speaker import calibrate
        calibrate()
    elif a.cmd == "run":
        from bud_e.ear import run
        run()
    elif a.cmd == "doctor":
        doctor()
    elif a.cmd == "profile":
        from bud_e.profile import summarise
        summarise()
    elif a.cmd == "latency":
        latency()
    elif a.cmd == "chat":
        chat(a.think)


def chat(think: bool) -> None:
    """The agent tier in a terminal. Prove tools work before using the mic."""
    from time import perf_counter

    from bud_e.agent import Agent
    agent = Agent()
    print("text chat with the agent tier. empty line to quit.\n")
    while True:
        q = input("you > ").strip()
        if not q:
            return
        t0, first = perf_counter(), None
        for kind, s in agent.turn(q, think=think):
            if kind == "tool":
                print(f"  [tool {s}]")
                continue
            first = first or perf_counter() - t0
            print(f"bud-e > {s}")
        print(f"  [first sentence {first or 0:.2f}s, "
              f"total {perf_counter() - t0:.2f}s]\n")


def _ttft(model: str, messages: list, tools=None, max_tokens: int = 40):
    """Seconds to first content or tool token, decode tok/s after it, and
    the prompt token count. Token counts come from the server's usage block,
    because oMLX can pack several tokens into one streamed chunk."""
    from time import perf_counter

    from bud_e.llm import local, sampling
    kw = sampling(False, max_tokens)
    if tools:
        kw["tools"] = tools
    t0, first, usage = perf_counter(), None, None
    for ch in local().chat.completions.create(
            model=model, messages=messages, stream=True,
            stream_options={"include_usage": True}, **kw):
        if getattr(ch, "usage", None):
            usage = ch.usage
        if ch.choices and (ch.choices[0].delta.content
                           or ch.choices[0].delta.tool_calls):
            first = first or perf_counter()
    end = perf_counter()
    n = usage.completion_tokens if usage else 0
    prompt = usage.prompt_tokens if usage else 0
    tps = (n - 1) / (end - first) if first and n > 1 and end > first else 0
    return (first or end) - t0, tps, prompt


def latency() -> None:
    """The three numbers that decide the architecture:
    cold vs warm time to first token (is the prefix cache hitting?),
    decode speed (is the quant right?), and whether the fast tier earns
    its memory."""
    import uuid

    from bud_e import config as C
    from bud_e.agent import Agent

    a = Agent()
    # A unique nonce at the end of the system prompt forces a genuinely cold
    # first call without touching the real prompt's cache entry.
    cold_sys = {"role": "system",
                "content": a.system["content"] + f"\n\n{uuid.uuid4()}"}
    user = {"role": "user", "content": "[test] Say hello in one sentence."}

    print("\nmeasuring\n")
    cold, _, ptok = _ttft(C.LOCAL_MODEL, [cold_sys, user], a.tools)
    warm, _, _ = _ttft(C.LOCAL_MODEL, [cold_sys, user], a.tools)
    _, tps, _ = _ttft(C.LOCAL_MODEL, [cold_sys, {
        "role": "user", "content": "Count from one to sixty in words."}],
        None, max_tokens=200)
    print(f"  agent prompt        {ptok:6d} tokens")
    print(f"  agent cold TTFT     {cold:6.2f}s   (system prompt + tools prefill)")
    print(f"  agent warm TTFT     {warm:6.2f}s   (same prefix again)")
    print(f"  agent decode        {tps:6.1f} tok/s  (speech needs ~4.5)")

    if warm > 0.6 * cold and cold > 1.0:
        print("\n  Warm is barely faster than cold: the prefix cache is not hitting.")
        print("  oMLX caches hybrid (Gated DeltaNet) models in 2048-token blocks;")
        print("  older builds skip any prompt shorter than one block. Look for")
        print("  'boundary_snapshot_unavailable' in ~/.omlx/logs/server.log and")
        print("  update oMLX (partial-tail caching, jundot/omlx #3835).")

    if C.FAST_TIER:
        try:
            _ttft(C.FAST_MODEL, [{"role": "user", "content": "hi"}])
            f_warm, _, _ = _ttft(C.FAST_MODEL, [{"role": "user", "content": "hi"}])
            print(f"  fast tier TTFT      {f_warm:6.2f}s")
        except Exception as e:
            print(f"  fast tier           FAILED {str(e)[:60]}")
            f_warm = None
        if warm < 0.8:
            print("\n  The 27B starts speaking in under 0.8s warm. The fast tier is")
            print("  not buying much: export BUD_E_FAST_TIER=0 and unload it.")
        elif f_warm is not None:
            print(f"\n  Fast tier saves {warm - f_warm:.2f}s on chat turns. Keep it.")
    print()


def doctor() -> None:
    import shutil

    from bud_e import config as C
    from bud_e import tools
    from bud_e.llm import list_models, local, sampling

    try:
        models = list_models()
        print(f"endpoint   {C.LOCAL_BASE}  ok ({len(models)} models)")
        want = [C.LOCAL_MODEL] + ([C.FAST_MODEL] if C.FAST_TIER else [])
        for m in want:
            print(f"model      {m}  {'ok' if m in models else 'NOT FOUND'}")
        if any(m not in models for m in want):
            print(f"           available: {', '.join(models)}")
    except Exception as e:
        print(f"endpoint   {C.LOCAL_BASE}  FAIL: {e}")
        print("           is oMLX running? Start it from the menu bar app; port comes from ~/.omlx/settings.json")
        return

    try:
        r = local().chat.completions.create(
            model=C.LOCAL_MODEL, messages=[{"role": "user", "content": "hi"}],
            **sampling(False, 40))
        msg = r.choices[0].message
        extra = getattr(msg, "model_extra", None) or {}
        leaked = "<think>" in (msg.content or "") or extra.get("reasoning_content")
        print(f"thinking   {'STILL ON: enable_thinking not honoured' if leaked else 'off'}")
    except Exception as e:
        print(f"thinking   check failed: {str(e)[:80]}")

    print(f"fast tier  {'on' if C.FAST_TIER else 'off'}")
    print(f"tools      {', '.join(tools.REGISTRY)}")
    print(f"icalBuddy  {shutil.which('icalBuddy') or 'not installed (brew install ical-buddy; AppleScript fallback misses recurring events)'}")
    print(f"web        {C.SEARXNG_URL or 'off (set BUD_E_SEARXNG to a local SearXNG)'}")
    print(f"hermes     {shutil.which('hermes') or 'not on PATH (delegate tool disabled)'}")
    print(f"read roots {', '.join(str(r) for r in C.READ_ROOTS)}")
    print(f"voiceprint {'yes' if C.PRINT_PATH.exists() else 'no'}")
    print(f"threshold  {C.THRESH_PATH.read_text().strip() if C.THRESH_PATH.exists() else 'not calibrated'}")


if __name__ == "__main__":
    main()
