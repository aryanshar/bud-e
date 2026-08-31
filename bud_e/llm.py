"""Two ways to reach a model: the raw local endpoint, and Hermes."""

import re
import subprocess
from datetime import datetime

from openai import OpenAI

from bud_e import config as C

ESCALATE = re.compile(r"^\s*ESCALATE\s*:?\s*(?P<what>.*)", re.I | re.S)

_client: OpenAI | None = None
_proc: subprocess.Popen | None = None


def local(base: str | None = None) -> OpenAI:
    """Raw endpoint. Used by the bench; the pipeline goes through Hermes."""
    global _client
    if _client is None or base:
        _client = OpenAI(base_url=base or C.LOCAL_BASE, api_key="local")
    return _client


def list_models(base: str | None = None) -> list[str]:
    return [m.id for m in local(base).models.list().data]


def hermes(prompt: str, *, skills: str | None = C.VOICE_SKILL,
           big: bool = False) -> str:
    """One agent turn, resuming the Bud-E session. Returns the final text."""
    cmd = ["hermes"]
    if C.HERMES_REASONING:
        cmd += ["--reasoning", C.HERMES_REASONING]
    cmd += ["chat", "--oneshot", "-Q",
           "-c", C.HERMES_SESSION, "--create-if-missing",
           "-t", C.HERMES_TOOLSETS,
           "--source", "tool",
           "-q", prompt + " /no_think"]   # soft switch rides the user turn
    if skills:
        cmd += ["-s", skills]
    if big:
        cmd += ["--provider", "anthropic", "--model", C.CLOUD_MODEL]

    global _proc
    try:
        _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        out, err = _proc.communicate(timeout=C.HERMES_TIMEOUT_S)
        rc = _proc.returncode
    except subprocess.TimeoutExpired:
        cancel()
        print(f"  [hermes timed out after {C.HERMES_TIMEOUT_S}s]")
        return "that took too long, I gave up on it."
    finally:
        _proc = None

    if rc < 0:                      # killed by cancel(); caller is discarding
        return ""

    # Older builds may not accept --reasoning. Drop it and retry once.
    if rc != 0 and "--reasoning" in cmd and (
            "unrecognized" in err.lower() or "invalid choice" in err.lower()):
        i = cmd.index("--reasoning")
        del cmd[i:i + 2]
        r2 = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=C.HERMES_TIMEOUT_S)
        out, err, rc = r2.stdout, r2.stderr, r2.returncode

    # Never speak stderr. A CLI usage message read aloud in a conversational
    # voice is worse than saying nothing useful at all.
    if rc != 0 or not out.strip():
        print(f"  [hermes failed rc={rc}] {(err or out).strip()[:300]}")
        return "something went wrong on my end."
    return out.strip()


def busy() -> bool:
    return _proc is not None and _proc.poll() is None


def cancel() -> None:
    """Kill an in-flight turn. Called when the user interrupts."""
    p = _proc
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(0.3)
        except subprocess.TimeoutExpired:
            p.kill()


def escalation_target(reply: str) -> str | None:
    """If the local model declined, return what it declined to answer."""
    m = ESCALATE.match(reply)
    return m.group("what").strip() if m else None


def log_escalation(text: str) -> None:
    """Record what the 27B could not handle. After a week of this you know
    whether an API tier is worth paying for, instead of guessing."""
    C.ensure_dirs()
    stamp = datetime.now().isoformat(timespec="seconds")
    with C.ESCALATION_LOG.open("a") as f:
        f.write(f"{stamp}\t{text}\n")
