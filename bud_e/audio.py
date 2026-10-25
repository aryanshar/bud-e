"""Microphone in (cut into utterances by Silero VAD), speech out (queued,
interruptible from any thread)."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np

from bud_e.config import Config

SR = 16_000
FRAME = 512                 # Silero wants exactly 512 samples at 16 kHz


def _device(spec: str):
    return int(spec) if spec.strip().isdigit() else (spec or None)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class Microphone:
    def __init__(self, cfg: Config) -> None:
        self.v = cfg.voice

    def record(self, seconds: float) -> np.ndarray:
        import sounddevice as sd
        buf = sd.rec(int(seconds * SR), samplerate=SR, channels=1,
                     dtype="float32", device=_device(self.v.input_device))
        sd.wait()
        return buf.reshape(-1)

    def utterances(self, on_onset: Callable[[float], None] | None = None
                   ) -> Iterator[np.ndarray]:
        """Yields one float32 array per thing said. Never stops listening.

        on_onset(loudness) fires the moment speech starts, long before the
        utterance ends. That is what lets you interrupt mid-sentence."""
        import sounddevice as sd
        import torch
        from silero_vad import VADIterator, load_silero_vad

        vad = VADIterator(load_silero_vad(), sampling_rate=SR,
                          threshold=self.v.vad_threshold,
                          min_silence_duration_ms=self.v.silence_ms)
        lead = 4                       # frames kept from just before onset
        recent: list[np.ndarray] = []
        buf: list[np.ndarray] = []
        talking = False
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=FRAME,
                            device=_device(self.v.input_device)) as stream:
            while True:
                chunk, _ = stream.read(FRAME)
                chunk = chunk.reshape(-1)
                event = vad(torch.from_numpy(chunk)) or {}
                recent = (recent + [chunk])[-lead:]

                if "start" in event and not talking:
                    talking, buf = True, list(recent)
                    if on_onset:
                        on_onset(rms(chunk))
                    continue
                if not talking:
                    continue
                buf.append(chunk)
                too_long = len(buf) * FRAME > self.v.max_speech_s * SR
                if "end" in event or too_long:
                    talking = False
                    vad.reset_states()
                    seg = np.concatenate(buf)
                    if len(seg) >= self.v.min_speech_s * SR:
                        yield seg


class Voice:
    """Say sentences one after another without gaps: synthesis happens on
    the caller's thread while the previous sentence plays on this one.
    stop() cuts everything off immediately."""

    CHUNK = 1024

    def __init__(self, engine, output_device: str = "") -> None:
        self.engine = engine
        self.device = _device(output_device)
        self.gen = 0                    # bumped by stop(); stale audio is dropped
        self._q: queue.Queue = queue.Queue()
        self._playing = False
        if engine is not None:
            threading.Thread(target=self._player, daemon=True).start()

    @property
    def speaking(self) -> bool:
        return self._playing or not self._q.empty()

    def say(self, text: str, cancel: threading.Event | None = None) -> None:
        if not text:
            return
        if self.engine is None:
            print(f"  [tts off] {text}")
            return
        gen = self.gen
        audio = self.engine.synth(text)
        if gen == self.gen and not (cancel and cancel.is_set()):
            self._q.put((gen, audio))

    def wait(self, cancel: threading.Event | None = None) -> None:
        while self.speaking and not (cancel and cancel.is_set()):
            time.sleep(0.02)

    def stop(self) -> None:
        self.gen += 1
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _player(self) -> None:
        import sounddevice as sd
        out = sd.OutputStream(samplerate=self.engine.sample_rate, channels=1,
                              dtype="float32", device=self.device)
        while True:
            gen, audio = self._q.get()
            if gen != self.gen:
                continue
            self._playing = True
            try:
                if not out.active:
                    out.start()
                for i in range(0, len(audio), self.CHUNK):
                    if gen != self.gen:
                        out.abort()          # drop what's buffered, now
                        break
                    out.write(audio[i:i + self.CHUNK].reshape(-1, 1))
                else:
                    if self._q.empty():
                        out.stop()           # waits for the buffer to drain
            except Exception as e:
                print(f"  [audio out failed: {e}]")
            finally:
                self._playing = False
