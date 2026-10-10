"""Microphone in, speech segments out, killable speech back to the speaker."""

import platform
import subprocess
import time
from collections.abc import Callable, Iterator

import numpy as np

from bud_e import config as C


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


class Speaker:
    """Interruptible speech. Kokoro (neural, local, MLX) when installed,
    macOS `say` otherwise. stop() from any thread cuts it mid-word."""

    def __init__(self) -> None:
        self._p: subprocess.Popen | None = None
        self.gen = 0            # bumped on stop; stream loops check it
        self._until = 0.0       # when current Kokoro audio finishes
        self._kokoro = None     # None = not tried, False = unavailable
        self._sr = 24_000

    # --- engines ---------------------------------------------------------
    def _engine(self):
        if C.TTS_ENGINE != "kokoro" or platform.system() != "Darwin":
            return False
        if self._kokoro is None:
            try:
                from mlx_audio.tts.utils import load_model
                self._kokoro = load_model(C.KOKORO_REPO)
                self._sr = getattr(self._kokoro, "sample_rate", 24_000)
            except Exception as e:
                print(f"  [kokoro unavailable, falling back to say: "
                      f"{str(e)[:120]}]")
                self._kokoro = False
        return self._kokoro

    def _synth(self, text: str) -> np.ndarray:
        parts = [np.asarray(r.audio, dtype=np.float32).reshape(-1)
                 for r in self._kokoro.generate(
                     text=text, voice=C.KOKORO_VOICE, speed=C.KOKORO_SPEED,
                     lang_code=C.KOKORO_LANG, verbose=False)]
        return np.concatenate(parts) if parts else np.zeros(1, np.float32)

    def warm(self) -> str:
        if self._engine():
            self._synth("Ready.")
            return f"kokoro {C.KOKORO_VOICE}"
        return f"say {C.TTS_VOICE}"

    # --- speaking --------------------------------------------------------
    def say(self, text: str, block: bool = False) -> None:
        if not text:
            return
        self._kill()            # replace current speech; not an interruption
        if platform.system() != "Darwin":
            print(f"[tts] {text}")
            return
        if self._engine():
            import sounddevice as sd
            gen = self.gen
            audio = self._synth(text)
            if gen != self.gen:         # interrupted while synthesising
                return
            sd.play(audio, self._sr)
            self._until = time.time() + len(audio) / self._sr
            if block:
                while time.time() < self._until and gen == self.gen:
                    time.sleep(0.02)
            return
        self._p = subprocess.Popen(
            ["say", "-v", C.TTS_VOICE, "-r", str(C.TTS_RATE), text[:800]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if block:
            self._p.wait()

    def stop(self) -> None:
        """Interrupt. Bumps gen so a streaming loop knows to stop."""
        self.gen += 1
        self._kill()

    def _kill(self) -> None:
        if self._until:
            import sounddevice as sd
            sd.stop()
            self._until = 0.0
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
        if self._until and time.time() < self._until:
            return True
        return self._p is not None and self._p.poll() is None


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd
    buf = sd.rec(int(seconds * C.SR), samplerate=C.SR, channels=1,
                 dtype="float32")
    sd.wait()
    return buf.flatten()


def listen(on_speech_start: Callable[[float], None] | None = None
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
                # The caller gates on loudness: Silero also fires on Bud-E's
                # own voice coming back through the speakers.
                if on_speech_start:
                    on_speech_start(rms(chunk))
            elif evt and "end" in evt and speaking:
                speaking = False
                vad.reset_states()
                seg = np.concatenate(buf)
                if len(seg) >= C.MIN_SPEECH_S * C.SR:
                    yield seg
