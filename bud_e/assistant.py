"""The voice loop.

  mic -> VAD -> [speaker check] -> Whisper -> wake word -> instant command
                                                         \\-> agent -> Kokoro

The microphone runs on the main thread and never blocks, so you can talk
over it: speech onset cancels the current turn and silences playback at
once. Turns run one at a time on a worker thread, each with its own cancel
flag.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bud_e import tts
from bud_e.agent import SAY, TOOL, Agent
from bud_e.asr import ASR
from bud_e.audio import Microphone, Voice
from bud_e.commands import instant
from bud_e.config import Config
from bud_e.speaker_id import SpeakerGate
from bud_e.turnlog import Turn


def wake_pattern(words: list[str]) -> re.Pattern:
    """'bud-e' also matches 'bud e', 'bude' and 'Bud-E,'."""
    alts = [r"[\s-]?".join(map(re.escape, re.split(r"[\s-]+", w)))
            for w in words if w.strip()]
    return re.compile(r"\b(" + "|".join(alts) + r")\b", re.I)


class Assistant:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.agent = Agent(cfg)
        self.asr = ASR(cfg)
        self.voice = Voice(tts.load(cfg), cfg.voice.output_device)
        self.gate = SpeakerGate.for_config(cfg)
        self.mic = Microphone(cfg)
        self.wake = wake_pattern(cfg.assistant.wake_words)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.cancel = threading.Event()      # the newest turn's flag
        self.busy = False
        self.echo = False                    # last onset was probably our voice
        self.last_exchange = 0.0

    # --- main thread ----------------------------------------------------
    def run(self) -> None:
        self.warm()
        print(f"\n{self.cfg.assistant.name} is listening. Say "
              f"\"{self.cfg.assistant.wake_words[0]}\" first. Ctrl-C to quit.\n")
        try:
            for seg in self.mic.utterances(on_onset=self.on_onset):
                self.on_utterance(seg)
        except KeyboardInterrupt:
            print()

    def warm(self) -> None:
        print("loading...")
        print(f"  model    {self.agent.llm.model} at {self.cfg.llm.base_url}")
        print(f"  tools    {', '.join(self.agent.tools.names) or 'none'}")
        print(f"  speech   {self.asr.warm()}")
        engine = self.voice.engine
        if engine is not None:
            engine.synth("Ready.")
        print(f"  voice    {type(engine).__name__ if engine else 'off (printing)'}")
        print(f"  speaker  {'on, threshold %.2f' % self.gate.threshold if self.gate else 'off (anyone can talk to it)'}")
        t = time.perf_counter()
        self.agent.warm()
        print(f"  prefill  {time.perf_counter() - t:.1f}s")

    def on_onset(self, level: float) -> None:
        v = self.cfg.voice
        talking = self.voice.speaking
        if level < (v.barge_rms_speaking if talking else v.barge_rms):
            # Quiet onset while it's talking is almost always its own voice
            # coming back through the speakers.
            self.echo = talking
            return
        self.echo = False
        if talking or self.busy:
            print(f"  [interrupted, level {level:.3f}]")
        self.cancel.set()
        self.voice.stop()

    def on_utterance(self, seg: np.ndarray) -> None:
        if self.echo:
            print("  [ignored: own voice]")
            return
        self.cancel = threading.Event()
        self.worker.submit(self._safe_turn, seg, self.cancel)

    # --- worker thread --------------------------------------------------
    def _safe_turn(self, seg: np.ndarray, cancel: threading.Event) -> None:
        self.busy = True
        try:
            self.handle(seg, cancel)
        except Exception as e:
            print(f"  [turn failed: {type(e).__name__}: {str(e)[:200]}]")
        finally:
            self.busy = False

    def handle(self, seg: np.ndarray, cancel: threading.Event) -> None:
        turn = Turn()
        if self.gate:
            with turn.stage("speaker"):
                score = self.gate.score(seg)
            if score < self.gate.threshold:
                print(f"  [ignored: speaker score {score:.2f}]")
                return

        with turn.stage("asr"):
            text = self.asr.transcribe(seg)
        if not text or cancel.is_set():
            return
        print(f"you > {text}")

        awake = time.time() - self.last_exchange < self.cfg.assistant.followup_seconds
        if not awake and not self.wake.search(text):
            print("  [no wake word]")
            return
        query = self.wake.sub("", text).strip(" ,.!?")
        if not query:
            self._say_all(["Yes?"], cancel)
            return

        reply = instant(query)
        if reply is not None:
            turn.meta["kind"] = "instant"
            self._say_all([reply] if reply else [], cancel, turn)
        else:
            turn.meta["kind"] = "agent"
            self._say_all(self._agent_sentences(query, cancel, turn), cancel,
                          turn)
        if not cancel.is_set():
            turn.save(self.cfg)
            print(turn.line() + "\n")

    def _agent_sentences(self, query, cancel, turn):
        tools = []
        for kind, value in self.agent.turn(query, cancel):
            if kind == TOOL:
                print(f"  [tool {value}]")
                tools.append(value)
                turn.meta["tools"] = tools
            elif kind == SAY:
                yield value

    def _say_all(self, sentences, cancel, turn: Turn | None = None) -> None:
        said = []
        for s in sentences:
            if cancel.is_set():
                break
            self.voice.say(s, cancel)
            if turn and not said:
                turn.mark("first_audio")
            said.append(s)
        if said:
            print(f"{self.cfg.assistant.name.lower()} > {' '.join(said)}")
        self.voice.wait(cancel)
        self.last_exchange = time.time()
