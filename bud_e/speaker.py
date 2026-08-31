"""Speaker verification. A filter against other voices, not a lock."""

import numpy as np

from bud_e import config as C
from bud_e.audio import record

_enc = None

ENROL_LINES = [
    "Bud-E, what does my calendar look like today",
    "Remind me to call the plumber on Tuesday afternoon",
    "The quick brown fox jumps over the lazy dog",
    "Bud-E, summarise the last thing I asked you about",
    "Six, seven, eight, nine, ten, eleven, twelve",
    "Turn the study lights off and lock the front door",
]


def encoder():
    global _enc
    if _enc is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(C.ECAPA_DIR))
    return _enc


def embed(wav: np.ndarray) -> np.ndarray:
    import torch
    e = encoder().encode_batch(torch.from_numpy(wav).unsqueeze(0))
    v = e.squeeze().detach().cpu().numpy()
    return v / np.linalg.norm(v)


def load_voiceprint() -> tuple[np.ndarray, float]:
    if not C.PRINT_PATH.exists():
        raise SystemExit("no voiceprint — run `bud-e enrol` first")
    thr = (float(C.THRESH_PATH.read_text())
           if C.THRESH_PATH.exists() else C.DEFAULT_THRESHOLD)
    return np.load(C.PRINT_PATH), thr


def enrol(n: int = 6) -> None:
    C.ensure_dirs()
    vecs = []
    for i in range(n):
        print(f"\n[{i+1}/{n}] say: {ENROL_LINES[i % len(ENROL_LINES)]}")
        input("press enter, then speak for ~4s > ")
        vecs.append(embed(record(4.0)))
        print("  captured")

    centroid = np.mean(vecs, axis=0)
    centroid /= np.linalg.norm(centroid)
    np.save(C.PRINT_PATH, centroid)

    scores = [float(v @ centroid) for v in vecs]
    print(f"\nsaved {C.PRINT_PATH}")
    print(f"self-similarity: min {min(scores):.3f}  mean {np.mean(scores):.3f}")
    print("now run: bud-e calibrate")


def calibrate(k: int = 4) -> None:
    centroid, _ = load_voiceprint()
    print("Record clips of YOU, then of anyone else "
          "(a housemate, a podcast, a video).")

    def batch(label: str) -> list[float]:
        out = []
        for i in range(k):
            input(f"{label} clip {i+1}/{k} — enter, then 4s of speech > ")
            out.append(float(embed(record(4.0)) @ centroid))
            print(f"  score {out[-1]:.3f}")
        return out

    me, other = batch("YOU"), batch("OTHER")
    lo, hi = min(me), max(other)
    print(f"\nyou:   min {lo:.3f}  mean {np.mean(me):.3f}")
    print(f"other: max {hi:.3f}  mean {np.mean(other):.3f}")

    if lo <= hi:
        print("\nDistributions overlap. Re-enrol in the room and on the mic "
              "you actually use, with longer clips. Do not lower the "
              "threshold by hand to paper over this.")
        return

    thr = (lo + hi) / 2
    C.THRESH_PATH.write_text(f"{thr:.4f}")
    print(f"\nthreshold {thr:.3f} written to {C.THRESH_PATH}")
