"""The loop.

Three tiers. Triggers answer in milliseconds. The fast model answers
conversation in about a second, streaming sentence by sentence so speech
starts before generation finishes. Hermes and the 27B handle anything that
needs tools, memory or real thought, and you wait for those on purpose.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bud_e import config as C
from bud_e import llm
from bud_e.asr import transcribe
from bud_e.audio import Speaker, listen
from bud_e.fast import Fast
from bud_e.llm import escalation_target, hermes, log_escalation
from bud_e.profile import Turn
from bud_e.router import needs_hermes
from bud_e.speaker import embed, encoder, load_voiceprint
from bud_e.triggers import match

WAKE = re.compile(r"\b(bud[\s-]?e|buddy|jarvis)\b", re.I)
FILLER = "one moment"
DECLINE = "that one's beyond me locally. I've noted it down."

_pool = ThreadPoolExecutor(max_workers=1)
_epoch = 0
voice = Speaker()
fast = Fast()


def barge_in() -> None:
    global _epoch
    if voice.speaking or llm.busy():
        _epoch += 1
        print("  [interrupted]")
    voice.stop()
    llm.cancel()


def slow_turn(text: str) -> str:
    """Hermes and the 27B. Tools, memory, skills, and the wait that buys."""
    reply = hermes(text)
    if not reply:
        return ""
    target = escalation_target(reply)
    if target is None:
        return reply

    log_escalation(target or text)
    if not C.CLOUD_ENABLED:
        print("  [declined, logged]")
        return DECLINE
    print("  [escalating]")
    voice.say(FILLER)
    return hermes(target or text, big=True)


def run() -> None:
    centroid, thr = load_voiceprint()
    print("warming models...")
    encoder()
    transcribe(np.zeros(C.SR, dtype="float32"))

    # Cold prefill of the 13k-token Hermes prompt costs ~25s. Pay it once
    # here rather than on the first thing you say.
    print("warming hermes (~25s the first time)...")
    t0 = time.time()
    hermes("hi")
    print(f"  prefill {time.time() - t0:.1f}s")

    print(f"bud-e listening (threshold {thr:.2f}). ctrl-c to stop.")
    last_turn = 0.0

    for seg in listen(on_speech_start=barge_in):
        epoch = _epoch
        T = Turn()

        with T.stage("embed"):
            score = float(embed(seg) @ centroid)
        if score < thr:
            print(f"  [ignored, speaker score {score:.2f}]")
            continue

        with T.stage("asr"):
            text = transcribe(seg)
        if not text:
            continue
        print(f"you ({score:.2f}) > {text}")

        awake = time.time() - last_turn < C.FOLLOWUP_WINDOW_S
        if not (awake or WAKE.search(text)):
            print("  [no wake phrase]")
            continue
        query = WAKE.sub("", text).strip(" ,.")

        # --- tier 0: triggers ------------------------------------------
        trigger, m = match(query)
        if trigger:
            with T.stage("llm"):
                reply = trigger.action(m)
            if reply:
                voice.say(reply)
                print(f"bud-e [trigger] > {reply}")
            T.meta = {"tier": "trigger"}
            T.save()
            print(T.line() + "\n")
            last_turn = time.time()
            continue

        # --- tier 1: fast model, streamed ------------------------------
        reason = needs_hermes(query)
        spoke = False
        if reason is None:
            with T.stage("llm"):
                said: list[str] = []
                routed = False
                try:
                    for kind, sentence in fast.turn(query):
                        if kind == "route":
                            routed = True
                            break
                        if epoch != _epoch:
                            break
                        said.append(sentence)
                        spoke = True
                        if not voice.say_now(sentence):
                            break
                except Exception as e:
                    print(f"  [fast tier failed: {str(e)[:80]}]")
                    routed = True
            if not routed and said:
                print(f"bud-e [fast] > {' '.join(said)}")
                T.meta = {"tier": "fast", "chars": len(" ".join(said))}
                T.save()
                print(T.line() + "\n")
                last_turn = time.time()
                continue
            reason = "fast declined"

        # --- tier 2: hermes + 27b --------------------------------------
        print(f"  [-> hermes: {reason}]")
        if not spoke:
            voice.say(FILLER)
        with T.stage("llm"):
            fut = _pool.submit(slow_turn, query)
            reply = fut.result()

        if epoch != _epoch or not reply:
            print("  [discarded, superseded]\n")
            continue

        with T.stage("tts_start"):
            voice.say(reply)
        print(f"bud-e [hermes] > {reply}")
        T.meta = {"tier": "hermes", "reason": reason, "chars": len(reply)}
        T.save()
        print(T.line() + "\n")
        last_turn = time.time()
