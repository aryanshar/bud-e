"""The loop.

The microphone is read on the main thread and never blocks, so speech onset
can interrupt Bud-E mid-sentence. Each utterance is handled on one worker
thread; a newer utterance bumps the epoch and the old one stops speaking and
stops generating.

Tiers:
  0  triggers   regex, no model, milliseconds
  1  fast       small model, no tools, streamed (optional, BUD_E_FAST_TIER)
  2  agent      the 27B direct: tools, memory, streamed, thinking on demand
"""

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bud_e import config as C
from bud_e import tools
from bud_e.agent import Agent
from bud_e.asr import transcribe
from bud_e.audio import Speaker, listen
from bud_e.fast import Fast
from bud_e.profile import Turn
from bud_e.router import route
from bud_e.speaker import embed, encoder, load_voiceprint
from bud_e.triggers import match

WAKE = re.compile(r"\b(bud[\s-]?e|buddy|jarvis)\b", re.I)
# Said the instant a turn needs a tool or thought, the way a person says
# "hang on" before they look something up. Varied so it never sounds canned.
CHECKING = ["Hang on.", "One sec.", "Let me look.", "Checking.", "Mm, one moment."]
THINKING = ["Hm, let me think about that.", "Give me a second on that one.",
            "Okay, thinking."]

_epoch = 0
_busy = threading.Event()
_pending: list[str] = []          # background results waiting for a gap
_last_turn = 0.0
_worker = ThreadPoolExecutor(max_workers=1)
voice = Speaker()
fast = Fast()
agent = Agent()


def barge_in(level: float) -> None:
    """Speech onset. While Bud-E talks, its own voice reaches the mic, so an
    interruption there has to be louder than the echo."""
    global _epoch
    if voice.speaking:
        if level < C.BARGE_RMS_SPEAKING:
            return
    elif level < C.BARGE_RMS:
        return
    if voice.speaking or _busy.is_set():
        _epoch += 1
        print(f"  [interrupted, level {level:.3f}]")
    voice.stop()


def announce(msg: str) -> None:
    """Called from a background Hermes job when it finishes."""
    print(f"  [background done] {msg[:120]}")
    if _busy.is_set() or voice.speaking:
        _pending.append(msg)
    else:
        voice.say(msg)


def speak_stream(gen, epoch: int, T: Turn, t0: float) -> tuple[list[str], bool]:
    """Speak sentences as they arrive. Returns (said, routed)."""
    said: list[str] = []
    try:
        for kind, s in gen:
            if epoch != _epoch:
                break
            if kind == "route":
                return said, True
            if kind == "tool":
                print(f"  [tool {s}]")
                continue
            if not said:
                T.meta["ttfa"] = round(time.perf_counter() - t0, 3)
            said.append(s)
            if not voice.say_now(s):
                break
    finally:
        gen.close()
    return said, False


def handle(seg: np.ndarray, epoch: int, centroid, thr: float) -> None:
    global _last_turn
    T = Turn()
    with T.stage("embed"):
        score = float(embed(seg) @ centroid)
    if score < thr:
        print(f"  [ignored, speaker score {score:.2f}]")
        return

    with T.stage("asr"):
        text = transcribe(seg)
    if not text or epoch != _epoch:
        return
    print(f"you ({score:.2f}) > {text}")

    awake = time.time() - _last_turn < C.FOLLOWUP_WINDOW_S
    if not (awake or WAKE.search(text)):
        print("  [no wake phrase]")
        return
    query = WAKE.sub("", text).strip(" ,.")
    if not query:
        voice.say(f"{C.ADDRESS.capitalize()}?")
        _last_turn = time.time()
        return

    _busy.set()
    try:
        _turn(query, epoch, T)
    finally:
        _busy.clear()
        _last_turn = time.time()
        while _pending and epoch == _epoch:
            voice.say_now(_pending.pop(0))


def _turn(query: str, epoch: int, T: Turn) -> None:
    # --- tier 0: triggers ----------------------------------------------
    trigger, m = match(query)
    if trigger:
        with T.stage("llm"):
            reply = trigger.action(m)
        if reply:
            voice.say(reply)
            print(f"bud-e [trigger] > {reply}")
        T.meta["tier"] = "trigger"
        T.save()
        print(T.line() + "\n")
        return

    tier, think, reason = route(query)
    t0 = time.perf_counter()

    # --- tier 1: fast model --------------------------------------------
    if tier == "fast":
        with T.stage("llm"):
            try:
                said, routed = speak_stream(fast.turn(query), epoch, T, t0)
            except Exception as e:
                print(f"  [fast tier failed: {str(e)[:80]}]")
                said, routed = [], True
        if not routed:
            if said:
                print(f"bud-e [fast] > {' '.join(said)}")
                agent.note(query, " ".join(said))
            T.meta.update(tier="fast", chars=len(" ".join(said)))
            T.save()
            print(T.line() + "\n")
            return
        reason = "fast declined"

    # --- tier 2: agent -------------------------------------------------
    print(f"  [-> agent: {reason}{', thinking' if think else ''}]")
    if think:
        voice.say(random.choice(THINKING))
    elif reason == "needs tools":
        voice.say(random.choice(CHECKING))
    with T.stage("llm"):
        try:
            said, _ = speak_stream(
                agent.turn(query, think=think,
                           cancelled=lambda: epoch != _epoch),
                epoch, T, t0)
        except Exception as e:
            print(f"  [agent failed: {str(e)[:160]}]")
            voice.say("something went wrong on my end.")
            said = []
    if epoch != _epoch:
        print("  [discarded, superseded]\n")
        return
    print(f"bud-e [agent] > {' '.join(said)}")
    if said:
        fast.note(query, " ".join(said))
    T.meta.update(tier="agent", reason=reason, think=think,
                  chars=len(" ".join(said)))
    T.save()
    print(T.line() + "\n")


def run() -> None:
    centroid, thr = load_voiceprint()
    tools.on_delegate_done = announce

    print("warming models...")
    encoder()
    transcribe(np.zeros(C.SR, dtype="float32"))
    print(f"  voice: {voice.warm()}")
    t0 = time.time()
    ptok = agent.warm()          # system prompt + tool schemas into the cache
    print(f"  agent prefill {time.time() - t0:.1f}s, prompt {ptok} tokens")
    if 0 < ptok < 2100:
        print("  [warning] agent prompt is under oMLX's 2048-token cache block;"
              " every turn will re-prefill. See bud_e/persona.py.")
    if C.FAST_TIER:
        try:
            next(fast.turn("hi"), None)
            fast.reset()
        except Exception as e:
            print(f"  [fast tier unavailable: {str(e)[:80]}]")

    print(f"bud-e listening (threshold {thr:.2f}, tools: "
          f"{', '.join(tools.REGISTRY)}). ctrl-c to stop.")

    def safe(seg, epoch):
        try:
            handle(seg, epoch, centroid, thr)
        except Exception as e:
            print(f"  [turn failed: {type(e).__name__}: {str(e)[:160]}]")

    for seg in listen(on_speech_start=barge_in):
        _worker.submit(safe, seg, _epoch)
