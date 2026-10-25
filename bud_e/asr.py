"""Speech to text with Whisper.

  mlx-whisper     Apple silicon, on the GPU. Default on macOS.
  faster-whisper  CTranslate2. On CUDA when there is a GPU, else CPU int8.
                  Default everywhere else.

~/.bud-e/vocab.txt is fed to Whisper as a prompt on every utterance. Adding
the names and jargon you actually say fixes more errors than a bigger model.
"""

from __future__ import annotations

import numpy as np

from bud_e.config import IS_MAC, Config


class ASR:
    def __init__(self, cfg: Config) -> None:
        v = cfg.voice
        self.backend = v.asr if v.asr != "auto" else (
            "mlx-whisper" if IS_MAC else "faster-whisper")
        self.model_name = v.whisper_model
        self.language = v.language or None
        self.prompt = self._vocab(cfg)
        self._fw = None
        self.device = ""

    @staticmethod
    def _vocab(cfg: Config) -> str:
        if not cfg.vocab_path.exists():
            cfg.ensure_home()
            cfg.vocab_path.write_text(f"{cfg.assistant.name}.\n")
        return cfg.vocab_path.read_text().strip()

    def transcribe(self, wav: np.ndarray) -> str:
        if self.backend == "mlx-whisper":
            return self._mlx(wav)
        if self.backend == "faster-whisper":
            return self._faster(wav)
        raise SystemExit(f"unknown asr backend {self.backend!r}")

    def warm(self) -> str:
        self.transcribe(np.zeros(16_000, dtype=np.float32))
        return f"{self.backend} {self.model_name}" + (
            f" on {self.device}" if self.device else "")

    def _mlx(self, wav: np.ndarray) -> str:
        import mlx_whisper
        repo = (self.model_name if "/" in self.model_name
                else f"mlx-community/whisper-{self.model_name}")
        r = mlx_whisper.transcribe(
            wav, path_or_hf_repo=repo, language=self.language,
            initial_prompt=self.prompt, condition_on_previous_text=False,
            temperature=0.0)
        return r["text"].strip()

    def _faster(self, wav: np.ndarray) -> str:
        if self._fw is None:
            import ctranslate2
            from faster_whisper import WhisperModel
            cuda = ctranslate2.get_cuda_device_count() > 0
            self.device = "cuda" if cuda else "cpu"
            self._fw = WhisperModel(self.model_name, device=self.device,
                                    compute_type="float16" if cuda else "int8")
        segs, _ = self._fw.transcribe(
            wav, language=self.language, initial_prompt=self.prompt,
            condition_on_previous_text=False, vad_filter=False, beam_size=1)
        return " ".join(s.text for s in segs).strip()
