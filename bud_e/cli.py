"""bud-e <command>. Run `bud-e -h` for the list."""

from __future__ import annotations

import argparse
import shutil
import time
from importlib import util
from pathlib import Path

from bud_e import config

EXAMPLE = Path(__file__).with_name("config.example.toml")


def _quiet_libraries() -> None:
    """torch, Hugging Face and friends print warnings at load time that look
    like errors but aren't. Hide them so real problems stand out."""
    import logging
    import os
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Both of these set their own log levels on import, so levels set here
    # get overridden: use HF's env var, and a filter for torch's macOS note.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects"
                      ).addFilter(lambda record: False)
    for name in ("speechbrain", "phonemizer"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_libraries()
    p = argparse.ArgumentParser(prog="bud-e", description=__doc__)
    p.add_argument("--config", type=Path, help="config file to use")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="write a starter ~/.bud-e/config.toml")
    sub.add_parser("doctor", help="check the model server, tools and audio")
    sub.add_parser("chat", help="talk to it in text (no audio needed)")
    sub.add_parser("run", help="start listening")
    sub.add_parser("enrol", help="record your voiceprint (speaker check)")
    sub.add_parser("calibrate", help="set the speaker check threshold")
    b = sub.add_parser("bench", help="score the model on tool use and speed")
    b.add_argument("--runs", type=int, default=2)
    b.add_argument("--report", action="store_true",
                   help="compare saved results instead of running")
    sub.add_parser("stats", help="timing per stage from logged voice turns")
    a = p.parse_args()

    if a.cmd == "init":
        return init(a.config)
    cfg = config.load(a.config)

    from openai import APIConnectionError
    try:
        dispatch(a, cfg)
    except APIConnectionError:
        raise SystemExit(f"\ncan't reach the model server at "
                         f"{cfg.llm.base_url}. Is it running? "
                         f"Try: bud-e doctor") from None


def dispatch(a: argparse.Namespace, cfg: config.Config) -> None:
    if a.cmd == "doctor":
        doctor(cfg)
    elif a.cmd == "chat":
        chat(cfg)
    elif a.cmd == "run":
        from bud_e.assistant import Assistant
        Assistant(cfg).run()
    elif a.cmd == "enrol":
        from bud_e.speaker_id import enrol
        enrol(cfg)
    elif a.cmd == "calibrate":
        from bud_e.speaker_id import calibrate
        calibrate(cfg)
    elif a.cmd == "bench":
        from bud_e import bench
        bench.report(cfg) if a.report else bench.run(cfg, a.runs)
    elif a.cmd == "stats":
        from bud_e.turnlog import stats
        stats(cfg)


def init(path: Path | None) -> None:
    path = path or config.config_path()
    if path.exists():
        raise SystemExit(f"{path} already exists; edit it directly")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXAMPLE, path)
    print(f"wrote {path}\nset [llm] base_url and model, then: bud-e doctor")


def chat(cfg: config.Config) -> None:
    from bud_e.agent import SAY, TOOL, Agent
    agent = Agent(cfg)
    print(f"{agent.llm.model} via {cfg.llm.base_url}")
    print(f"tools: {', '.join(agent.tools.names) or 'none'}")
    print("empty line to quit, /reset to forget the conversation\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        if q == "/reset":
            agent.reset()
            continue
        t0, first = time.perf_counter(), None
        for kind, value in agent.turn(q):
            if kind == TOOL:
                print(f"  [tool {value}]")
            elif kind == SAY:
                first = first or time.perf_counter() - t0
                print(f"{cfg.assistant.name.lower()} > {value}")
        print(f"  [first sentence {first or 0:.2f}s, "
              f"done {time.perf_counter() - t0:.2f}s]\n")


def _row(label: str, ok: bool | None, detail: str) -> None:
    mark = {True: "ok  ", False: "FAIL", None: "--  "}[ok]
    print(f"  {mark} {label:<13} {detail}")


def doctor(cfg: config.Config) -> None:
    from bud_e.llm import LLM
    from bud_e.tools import Toolbox

    print(f"\nconfig  {cfg.source or 'none found, using defaults (bud-e init)'}")
    print(f"state   {cfg.home}\n")

    llm = LLM(cfg)
    try:
        models = llm.models()
        _row("server", True, f"{cfg.llm.base_url} ({len(models)} models)")
    except Exception as e:
        _row("server", False, f"{cfg.llm.base_url}: {str(e)[:100]}")
        print("\n  Start your model server first. See README, step 1.\n")
        return

    if cfg.llm.model and cfg.llm.model not in models:
        _row("model", False, f"{cfg.llm.model} not served. "
             f"Available: {', '.join(models[:8])}")
        return
    note = "" if cfg.llm.model else "  (first listed; set [llm] model)"
    _row("model", True, llm.model + note)

    toolbox = Toolbox(cfg)
    probe = toolbox.schemas()[:1] or None
    try:
        t0 = time.perf_counter()
        events = list(llm.stream(
            [{"role": "user", "content": "Save this fact: I like tea."}],
            probe, max_tokens=200))
        dt = time.perf_counter() - t0
        text = "".join(v for k, v in events if k == "text")
        calls = events[-1][1]
        _row("reply", True, f"{dt:.2f}s for a short tool turn")
        _row("tool calls", bool(calls),
             "working" if calls else "model answered in text instead; "
             "check the server's tool-call support (README, troubleshooting)")
        _row("thinking", "<think>" not in text,
             "off" if "<think>" not in text else
             "model is thinking out loud; hidden, but it adds delay")
    except Exception as e:
        _row("reply", False, str(e)[:160])

    _row("tools", True, ", ".join(toolbox.names))
    if config.IS_MAC:
        _row("icalBuddy", bool(shutil.which("icalBuddy")) or None,
             shutil.which("icalBuddy") or "not installed: brew install "
             "ical-buddy (fallback misses recurring events)")

    print()
    voice_mods = {"sounddevice": "audio I/O", "silero_vad": "voice detection",
                  "kokoro": "neural voice",
                  "mlx_whisper" if config.IS_MAC else "faster_whisper":
                  "speech recognition", "speechbrain": "speaker check"}
    for mod, what in voice_mods.items():
        found = util.find_spec(mod) is not None
        optional = mod in ("speechbrain", "kokoro")
        _row(mod, True if found else (None if optional else False),
             what + ("" if found else " - not installed "
                     "(uv sync --extra voice" +
                     (" --extra speaker" if mod == "speechbrain" else "") + ")"))
    if util.find_spec("sounddevice"):
        try:
            import sounddevice as sd
            i, o = sd.query_devices(kind="input"), sd.query_devices(kind="output")
            _row("mic", True, i["name"])
            _row("speaker", True, o["name"])
        except Exception as e:
            _row("audio", False, str(e)[:100])
    enrolled = cfg.voiceprint_path.exists()
    _row("voiceprint", True if enrolled else None,
         "enrolled" if enrolled else "not enrolled (optional: bud-e enrol)")
    print()


if __name__ == "__main__":
    main()
