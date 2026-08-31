import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="bud-e")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enrol", help="record clips and build your voiceprint")
    sub.add_parser("calibrate", help="score yourself vs others, set threshold")
    sub.add_parser("run", help="start the listener")
    sub.add_parser("doctor", help="check the local endpoint and hermes")
    sub.add_parser("profile", help="per-stage timing across logged turns")
    sub.add_parser("latency", help="decompose LLM latency: CLI vs model")
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


def latency() -> None:
    """Split LLM time into process startup, Hermes overhead, and the model.

    Worth knowing before you optimise the wrong one: `hermes chat --oneshot`
    spawns a process per turn, and that cost is fixed no matter how fast the
    model gets.
    """
    import subprocess
    from time import perf_counter

    from bud_e import config as C
    from bud_e.llm import hermes, local

    def timed(label, fn):
        t0 = perf_counter()
        try:
            fn()
            dt = perf_counter() - t0
            print(f"  {label:<22} {dt:6.2f}s")
            return dt
        except Exception as e:
            print(f"  {label:<22}  FAILED  {str(e)[:60]}")
            return None

    print("\nmeasuring, this takes about a minute\n")

    spawn = timed("hermes startup", lambda: subprocess.run(
        ["hermes", "--version"], capture_output=True))

    raw = timed("model direct", lambda: local().chat.completions.create(
        model=C.LOCAL_MODEL, max_tokens=60,
        messages=[{"role": "user", "content": "Say hello in one sentence. /no_think"}]))

    full = timed("hermes full turn", lambda: hermes("Say hello in one sentence."))

    if None in (spawn, raw, full):
        return
    overhead = full - raw
    print("\n  ---")
    print(f"  process spawn      {spawn:6.2f}s  ({spawn/full:.0%} of a turn)")
    print(f"  model generation   {raw:6.2f}s  ({raw/full:.0%})")
    print(f"  hermes overhead    {overhead - spawn:6.2f}s  "
          f"(tools, skills, session)")
    if spawn / full > 0.25:
        print("\n  Spawn is over a quarter of every turn. `hermes serve` and")
        print("  an HTTP call would remove it entirely.")
    print()


def doctor() -> None:
    import shutil
    import subprocess

    from bud_e import config as C
    from bud_e.llm import list_models

    try:
        models = list_models()
        ok = C.LOCAL_MODEL in models
        print(f"endpoint  {C.LOCAL_BASE}  ok ({len(models)} models)")
        print(f"model     {C.LOCAL_MODEL}  {'loaded' if ok else 'NOT LOADED'}")
        if not ok:
            print(f"          available: {', '.join(models)}")
    except Exception as e:
        print(f"endpoint  {C.LOCAL_BASE}  FAIL: {e}")

    print(f"hermes    {shutil.which('hermes') or 'NOT ON PATH'}")
    if shutil.which("hermes"):
        r = subprocess.run(["hermes", "config", "get", "model.default"],
                           capture_output=True, text=True)
        print(f"          model.default = {r.stdout.strip()}")

    print(f"voiceprint {'yes' if C.PRINT_PATH.exists() else 'no'}")
    print(f"threshold  {C.THRESH_PATH.read_text().strip()
                        if C.THRESH_PATH.exists() else 'not calibrated'}")
    print(f"cloud tier {'on -> ' + C.CLOUD_MODEL if C.CLOUD_ENABLED
                        else 'off (local only)'}")
    if C.ESCALATION_LOG.exists():
        n = sum(1 for _ in C.ESCALATION_LOG.open())
        print(f"declined   {n} turns logged in {C.ESCALATION_LOG}")


if __name__ == "__main__":
    main()
