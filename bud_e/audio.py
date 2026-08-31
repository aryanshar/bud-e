"""Microphone in, speech segments out, killable speech back to the speaker."""

import platform
import subprocess
from collections.abc import Callable, Iterator

import numpy as np

from bud_e import config as C


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


class Speaker:
    """Non-blocking TTS you can cut off mid-sentence."""

    def __init__(self) -> None:
        self._p: subprocess.Popen | None = None
        self.gen = 0            # bumped on stop; stream loops check it

    def say(self, text: str, block: bool = False) -> None:
        if not text:
            return
        self.stop()
        if platform.system() != "Darwin":
            print(f"[tts] {text}")
            return
        self._p = subprocess.Popen(
            ["say", "-v", C.TTS_VOICE, "-r", str(C.TTS_RATE), text[:800]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if block:
            self._p.wait()

    def stop(self) -> None:
        self.gen += 1
        if self._p and self._p.poll() is None:
            self._p.terminate()
            try:
                self._p.wait(0.3)
            except subprocess.TimeoutExpired:
                self._p.kill()
        self._p = None

    def say_now(self, text: str) -> bool:
        """Speak and wait. Returns False if interrupted partway."""
        gen = self.gen
        self.say(text, block=True)
        return gen == self.gen

    @property
    def speaking(self) -> bool:
        return self._p is not None and self._p.poll() is None


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd
    buf = sd.rec(int(seconds * C.SR), samplerate=C.SR, channels=1,
                 dtype="float32")
    sd.wait()
    return buf.flatten()


def listen(on_speech_start: Callable[[], None] | None = None
           ) -> Iterator[np.ndarray]:
    """Yield float32 speech segments. Never stops listening, so the caller
    can be interrupted mid-reply.

    on_speech_start fires the instant Silero detects speech, before the
    utterance has finished. That is what makes barge-in feel immediate: you
    cannot wait for a full segment and still call it an interruption.
    """
    import sounddevice as sd
    import torch
    from silero_vad import VADIterator, load_silero_vad

    vad = VADIterator(load_silero_vad(), sampling_rate=C.SR,
                      threshold=C.VAD_THRESHOLD,
                      min_silence_duration_ms=C.VAD_SILENCE_MS)
    buf: list[np.ndarray] = []
    speaking = False

    with sd.InputStream(samplerate=C.SR, channels=1, dtype="float32",
                        blocksize=C.FRAME) as stream:
        while True:
            chunk, _ = stream.read(C.FRAME)
            chunk = chunk.flatten()
            evt = vad(torch.from_numpy(chunk))

            if speaking:
                buf.append(chunk)
                if len(buf) * C.FRAME > C.MAX_SPEECH_S * C.SR:
                    evt = {"end": 0}

            if evt and "start" in evt:
                speaking, buf = True, [chunk]
                # Loudness gate. Silero fires on the assistant's own voice
                # coming back through the speakers; requiring real level
                # keeps that from self-interrupting.
                if on_speech_start and rms(chunk) >= C.BARGE_RMS:
                    on_speech_start()
            elif evt and "end" in evt and speaking:
                speaking = False
                vad.reset_states()
                seg = np.concatenate(buf)
                if len(seg) >= C.MIN_SPEECH_S * C.SR:
                    yield seg
