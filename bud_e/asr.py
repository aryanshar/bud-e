"""Speech to text, with a vocabulary prompt that matters more than the model.

Backends, set with BUD_E_ASR:
  mlx-whisper  (default) accurate, ~0.4s on a 5s clip
  parakeet     faster, English only, weaker on accented speech
  faster       CPU fallback for non-Darwin
"""

import os
import platform

import numpy as np

from bud_e import config as C

_fw = None
_parakeet = None
_prompt: str | None = None


def vocabulary() -> str:
    """Biasing Whisper with your own words fixes more accent and proper-noun
    errors than swapping models does. Edit ~/.bud-e/vocab.txt to extend it."""
    global _prompt
    if _prompt is not None:
        return _prompt
    C.ensure_dirs()
    if not C.VOCAB_PATH.exists():
        C.VOCAB_PATH.write_text(
            "Bud-E, Hermes, LM Studio, Qwen, Anthropic, Bengaluru, JLR, "
            "LiDAR, GroundingDINO, Kubernetes, PyTorch, repo, endpoint, "
            "quantisation, latency, inference.\n")
    _prompt = C.VOCAB_PATH.read_text().strip()
    return _prompt


def _mlx(wav: np.ndarray) -> str:
    import mlx_whisper
    r = mlx_whisper.transcribe(
        wav,
        path_or_hf_repo=C.WHISPER_REPO,
        language=C.ASR_LANGUAGE,
        initial_prompt=vocabulary(),
        condition_on_previous_text=False,   # stops drift across utterances
        temperature=0.0,
        fp16=True)
    return r["text"].strip()


def _parakeet_asr(wav: np.ndarray) -> str:
    global _parakeet
    from parakeet_mlx import from_pretrained
    if _parakeet is None:
        _parakeet = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
    return _parakeet.transcribe(wav).text.strip()


def _faster(wav: np.ndarray) -> str:
    global _fw
    from faster_whisper import WhisperModel
    if _fw is None:
        _fw = WhisperModel("large-v3-turbo", compute_type="int8")
    segs, _ = _fw.transcribe(wav, language=C.ASR_LANGUAGE,
                             initial_prompt=vocabulary(),
                             condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


def transcribe(wav: np.ndarray) -> str:
    backend = os.environ.get("BUD_E_ASR",
                             "mlx-whisper" if platform.system() == "Darwin"
                             else "faster")
    if backend == "parakeet":
        return _parakeet_asr(wav)
    if backend == "mlx-whisper":
        return _mlx(wav)
    return _faster(wav)
