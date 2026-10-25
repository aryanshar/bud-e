"""Text to speech. Each engine turns a sentence into a float32 array.

  kokoro  neural, local, sounds human. Runs on CUDA if present, else CPU,
          which on Apple silicon is still several times faster than speech.
  say     the macOS built-in voice. Instant, robotic, zero install.
  none    print instead of speaking (headless boxes, debugging).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bud_e.config import IS_MAC, Config


class Kokoro:
    sample_rate = 24_000

    def __init__(self, voice: str, speed: float) -> None:
        import torch
        from kokoro import KPipeline
        self.voice, self.speed = voice, speed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Voice names start with their accent: a = American, b = British.
        self.pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M",
                              device=device)

    def synth(self, text: str) -> np.ndarray:
        parts = []
        for _, _, audio in self.pipe(text, voice=self.voice, speed=self.speed):
            if audio is None:
                continue
            if hasattr(audio, "cpu"):            # torch tensor, maybe on GPU
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(1, np.float32)


class MacSay:
    sample_rate = 24_000

    def __init__(self, voice: str) -> None:
        self.voice = voice

    def synth(self, text: str) -> np.ndarray:
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "say.wav"
            subprocess.run(["say", "-v", self.voice, "-o", str(path),
                            "--file-format=WAVE",
                            f"--data-format=LEF32@{self.sample_rate}",
                            text[:1000]], check=True, timeout=30)
            audio, _ = sf.read(path, dtype="float32")
        return audio.reshape(-1)


def load(cfg: Config):
    """The configured engine, or None for print-only."""
    v = cfg.voice
    choice = v.tts
    if choice in ("auto", "kokoro"):
        try:
            return Kokoro(v.kokoro_voice, v.kokoro_speed)
        except Exception as e:
            if choice == "kokoro":
                raise
            print(f"  [kokoro unavailable: {str(e)[:120]}]")
    if choice in ("auto", "say") and IS_MAC:
        return MacSay(v.say_voice)
    if choice not in ("auto", "none"):
        raise SystemExit(f"tts = {choice!r} is not available here")
    return None
