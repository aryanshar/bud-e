"""Optional: only respond to your voice (ECAPA speaker embeddings).

A filter, not a lock. It keeps a housemate or the TV out; a recording of you
gets through. Nothing dangerous should rely on it.

  bud-e enrol       six short clips of you -> ~/.bud-e/voiceprint.npy
  bud-e calibrate   you vs other voices    -> ~/.bud-e/threshold.txt
"""

from __future__ import annotations

import numpy as np

from bud_e.audio import Microphone
from bud_e.config import Config

DEFAULT_THRESHOLD = 0.35
LINES = [
    "Bud-E, what does my calendar look like today",
    "Remind me to call the plumber on Tuesday afternoon",
    "The quick brown fox jumps over the lazy dog",
    "Bud-E, summarise the last thing I asked you about",
    "Six, seven, eight, nine, ten, eleven, twelve",
    "Turn the study lights off and lock the front door",
]


class SpeakerGate:
    def __init__(self, cfg: Config) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(cfg.home / "ecapa"))
        self.print = (np.load(cfg.voiceprint_path)
                      if cfg.voiceprint_path.exists() else None)
        self.threshold = (float(cfg.threshold_path.read_text())
                          if cfg.threshold_path.exists() else DEFAULT_THRESHOLD)

    @classmethod
    def for_config(cls, cfg: Config) -> SpeakerGate | None:
        """None when the check is off, or 'auto' and you haven't enrolled."""
        mode = cfg.voice.speaker_check
        if mode == "off" or (mode == "auto"
                             and not cfg.voiceprint_path.exists()):
            return None
        try:
            gate = cls(cfg)
        except ImportError:
            if mode == "on":
                raise SystemExit("speaker_check = on needs the speaker extra: "
                                 "uv sync --extra voice --extra speaker")
            print("  [speaker check skipped: speechbrain not installed]")
            return None
        if gate.print is None:
            raise SystemExit("speaker_check = on but no voiceprint: "
                             "run `bud-e enrol`")
        return gate

    def embed(self, wav: np.ndarray) -> np.ndarray:
        import torch
        e = self.encoder.encode_batch(torch.from_numpy(wav).unsqueeze(0))
        v = e.squeeze().detach().cpu().numpy()
        return v / np.linalg.norm(v)

    def score(self, wav: np.ndarray) -> float:
        return float(self.embed(wav) @ self.print)


def enrol(cfg: Config, n: int = 6) -> None:
    cfg.ensure_home()
    gate, mic = SpeakerGate(cfg), Microphone(cfg)
    vecs = []
    for i in range(n):
        print(f"\n[{i + 1}/{n}] say: {LINES[i % len(LINES)]}")
        input("press enter, then speak for about 4 seconds > ")
        vecs.append(gate.embed(mic.record(4.0)))
    centroid = np.mean(vecs, axis=0)
    centroid /= np.linalg.norm(centroid)
    np.save(cfg.voiceprint_path, centroid)
    scores = [float(v @ centroid) for v in vecs]
    print(f"\nsaved {cfg.voiceprint_path}")
    print(f"self-similarity min {min(scores):.3f}, mean {np.mean(scores):.3f}")
    print("next: bud-e calibrate")


def calibrate(cfg: Config, k: int = 4) -> None:
    gate, mic = SpeakerGate(cfg), Microphone(cfg)
    if gate.print is None:
        raise SystemExit("no voiceprint yet: run `bud-e enrol` first")
    print("Record clips of you, then of anyone else (a housemate, a podcast).")

    def batch(label: str) -> list[float]:
        out = []
        for i in range(k):
            input(f"{label} clip {i + 1}/{k}: enter, then 4s of speech > ")
            out.append(gate.score(mic.record(4.0)))
            print(f"  score {out[-1]:.3f}")
        return out

    me, other = batch("YOU"), batch("OTHER")
    print(f"\nyou   min {min(me):.3f}  mean {np.mean(me):.3f}")
    print(f"other max {max(other):.3f}  mean {np.mean(other):.3f}")
    if min(me) <= max(other):
        print("\nThe two overlap. Re-enrol in the room and on the mic you "
              "actually use, rather than lowering the threshold by hand.")
        return
    thr = (min(me) + max(other)) / 2
    cfg.threshold_path.write_text(f"{thr:.4f}")
    print(f"\nthreshold {thr:.3f} saved to {cfg.threshold_path}")
