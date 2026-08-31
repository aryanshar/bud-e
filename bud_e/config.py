"""Every tunable lives here. The bench and the pipeline read the same values."""

import os
from pathlib import Path

# --- inference ---------------------------------------------------------
LOCAL_BASE = os.environ.get("BUD_E_BASE", "http://localhost:1234/v1")
LOCAL_MODEL = os.environ.get("BUD_E_MODEL", "qwen3.8-27b-uncensored-mlx")
CLOUD_MODEL = os.environ.get("BUD_E_CLOUD_MODEL", "claude-sonnet-4-6")

# Fast tier. Answers conversation directly: no Hermes, no tools, ~200 token
# prompt, so there is no prefill to pay. Load it in LM Studio alongside the
# 27B; together they fit comfortably in 48GB.
FAST_MODEL = os.environ.get("BUD_E_FAST_MODEL", "google/gemma-4-e4b")
FAST_MAX_TOKENS = int(os.environ.get("BUD_E_FAST_MAX_TOKENS", "160"))
FAST_HISTORY = int(os.environ.get("BUD_E_FAST_HISTORY", "8"))   # message pairs
ROUTE_TOKEN = "ROUTE"

# Escalation is off unless a key exists AND you opt in. Local-only is the
# default, and nothing in the pipeline breaks without it.
CLOUD_ENABLED = (
    os.environ.get("BUD_E_CLOUD", "0") == "1"
    and bool(os.environ.get("ANTHROPIC_API_KEY"))
)

# --- hermes ------------------------------------------------------------
HERMES_SESSION = os.environ.get("BUD_E_SESSION", "bud-e-voice")
HERMES_TOOLSETS = os.environ.get("BUD_E_TOOLSETS", "file,web,skills,memory")
VOICE_SKILL = "bud-e-voice"
HERMES_TIMEOUT_S = 180
# Hermes global --reasoning flag. Empty string omits it entirely.
HERMES_REASONING = os.environ.get("BUD_E_REASONING", "low")

# --- audio -------------------------------------------------------------
SR = 16_000
FRAME = 512                  # Silero requires exactly 512 samples at 16 kHz
MIN_SPEECH_S = 1.2           # below this, speaker verification is a coin flip
MAX_SPEECH_S = 20.0
FOLLOWUP_WINDOW_S = 30       # wake phrase not needed inside this window
DEFAULT_THRESHOLD = 0.35     # placeholder only; calibrate overwrites it

VAD_THRESHOLD = float(os.environ.get("BUD_E_VAD", "0.5"))
VAD_SILENCE_MS = int(os.environ.get("BUD_E_SILENCE_MS", "380"))
BARGE_RMS = float(os.environ.get("BUD_E_BARGE_RMS", "0.03"))

TTS_VOICE = os.environ.get("BUD_E_VOICE", "Daniel")
TTS_RATE = int(os.environ.get("BUD_E_RATE", "195"))

# --- persona -----------------------------------------------------------
ADDRESS = os.environ.get("BUD_E_ADDRESS", "sir")
PERSONA = os.environ.get("BUD_E_PERSONA", "").strip()

# --- asr ---------------------------------------------------------------
WHISPER_REPO = os.environ.get("BUD_E_WHISPER",
                              "mlx-community/whisper-large-v3-turbo")
ASR_LANGUAGE = os.environ.get("BUD_E_LANG", "en")

# --- state -------------------------------------------------------------
HOME = Path(os.environ.get("BUD_E_HOME", Path.home() / ".bud-e"))
PRINT_PATH = HOME / "voiceprint.npy"
THRESH_PATH = HOME / "threshold.txt"
ECAPA_DIR = HOME / "ecapa"
ESCALATION_LOG = HOME / "escalations.log"
TURN_LOG = HOME / "turns.jsonl"
VOCAB_PATH = HOME / "vocab.txt"

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def ensure_dirs() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
