"""The voice loop.

  mic -> VAD -> [speaker check] -> Whisper -> wake word -> instant command
                                                         \\-> agent -> Kokoro

The microphone runs on the main thread and never blocks, so you can talk
over it: speech onset cancels the current turn and silences playback at
once. Turns run one at a time on a worker thread, each with its own cancel
flag.
"""

from __future__ import annotations

import random
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


LOOKUPS = {"get_weather", "get_calendar", "list_reminders", "web_search",
           "find_files", "read_file", "read_notes", "recall"}
FILLERS = ["One sec.", "Let me check.", "Hang on.", "Checking."]


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
        self.last_exchange = 0.0

    # --- main thread ----------------------------------------------------
    def run(self) -> None:
        self.warm()
        print(f"\n{self.cfg.assistant.name} is listening. Say "
              f"\"{self.cfg.assistant.wake_words[0]}\" first. Ctrl-C to quit.\n")
        try:
            for seg in self.mic.utterances(
                    on_onset=self.on_onset,
                    is_speaking=lambda: self.voice.speaking,
                    on_barge=self.on_barge):
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
        n = self.agent.warm()
        print(f"  prefill  {time.perf_counter() - t:.1f}s"
              + (f", prompt {n} tokens" if n else ""))

    def on_onset(self, level: float) -> None:
        """You started talking while it was quiet. If a turn is still
        thinking, a new request replaces it."""
        if self.busy and level >= self.cfg.voice.barge_rms:
            print(f"  [interrupted while thinking, level {level:.3f}]")
            self.cancel.set()

    def on_barge(self, level: float) -> None:
        """You talked over it, loud enough to beat its own echo."""
        print(f"  [interrupted, level {level:.3f}]")
        self.cancel.set()
        self.voice.stop()

    def on_utterance(self, seg: np.ndarray) -> None:
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
        seconds = len(seg) / 16_000
        # Voice checks on very short clips are close to a coin flip, so a
        # bare "buddy" goes through; the request that follows gets checked.
        if self.gate and seconds >= self.cfg.voice.speaker_check_min_s:
            with turn.stage("speaker"):
                score = self.gate.score(seg)
            if score < self.gate.threshold:
                print(f"  [ignored: speaker score {score:.2f} < "
                      f"{self.gate.threshold:.2f}, {seconds:.1f}s clip]")
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
            # Just the wake word: chime instead of talking over what you
            # say next.
            self.voice.chime()
            self.last_exchange = time.time()
            print("  [listening]")
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
        tools, spoke = [], False
        for kind, value in self.agent.turn(query, cancel):
            if kind == TOOL:
                print(f"  [tool {value}]")
                tools.append(value)
                turn.meta["tools"] = tools
                # A lookup means a second model call before the answer, so
                # say something now, like a person would before checking.
                if not spoke and value in LOOKUPS:
                    spoke = True
                    yield random.choice(FILLERS)
            elif kind == SAY:
                spoke = True
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
