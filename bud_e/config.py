"""Every tunable lives here. The bench and the pipeline read the same values."""

import os
from pathlib import Path


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# --- inference (oMLX) ----------------------------------------------------
# Port and API key come from oMLX's own settings so they never drift. The
# model id is the folder name under ~/.omlx/models; `bud-e doctor` lists them.
OMLX_SETTINGS = Path.home() / ".omlx" / "settings.json"


def _omlx() -> dict:
    try:
        import json
        return json.loads(OMLX_SETTINGS.read_text())
    except Exception:
        return {}


_om = _omlx()
_port = _om.get("server", {}).get("port", 8000)
LOCAL_BASE = os.environ.get("BUD_E_BASE", f"http://localhost:{_port}/v1")
LOCAL_MODEL = os.environ.get("BUD_E_MODEL", "Qwen3.8-27B-MLX-4bit")
API_KEY = (os.environ.get("BUD_E_API_KEY")
           or _om.get("auth", {}).get("api_key") or "local")

# Qwen's published sampling. Non-thinking for normal turns, thinking for the
# turns the router marks as depth. Sent per request so the server's per-model
# defaults cannot silently drift.
SAMPLING = {
    False: dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                presence_penalty=float(os.environ.get("BUD_E_PRESENCE", "1.5"))),
    True: dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
               presence_penalty=0.0),
}

# --- agent tier (27B, direct, streamed, native tool calls) ---------------
AGENT_MAX_TOKENS = int(os.environ.get("BUD_E_MAX_TOKENS", "512"))
AGENT_THINK_MAX_TOKENS = int(os.environ.get("BUD_E_THINK_MAX_TOKENS", "4096"))
AGENT_MAX_STEPS = 5          # tool round trips per turn; local models loop
AGENT_HISTORY = int(os.environ.get("BUD_E_HISTORY", "12"))     # turns kept
TOOL_RESULT_CHARS = 6000     # long tool output costs prefill on every step

# --- fast tier (optional) ------------------------------------------------
# Small model, ~200 token prompt, no tools. Only worth keeping if the 27B's
# warm time-to-first-token is above ~0.8s. `bud-e latency` measures it.
FAST_TIER = _env_bool("BUD_E_FAST_TIER", "1")
FAST_MODEL = os.environ.get("BUD_E_FAST_MODEL", "gemma-4-E4B-it-MLX-4bit")
FAST_MAX_TOKENS = int(os.environ.get("BUD_E_FAST_MAX_TOKENS", "160"))
FAST_HISTORY = int(os.environ.get("BUD_E_FAST_HISTORY", "8"))   # message pairs
ROUTE_TOKEN = "ROUTE"

# --- tools ---------------------------------------------------------------
# read_file and find_files never leave these roots.
READ_ROOTS = [Path(p).expanduser() for p in os.environ.get(
    "BUD_E_READ_ROOTS", "~/Documents:~/Desktop:~/Downloads").split(":") if p]
WEATHER_LAT = float(os.environ.get("BUD_E_LAT", "12.9716"))
WEATHER_LON = float(os.environ.get("BUD_E_LON", "77.5946"))
WEATHER_PLACE = os.environ.get("BUD_E_PLACE", "Bengaluru")
# Local SearXNG instance, e.g. http://localhost:8888. Empty = no web tool.
SEARXNG_URL = os.environ.get("BUD_E_SEARXNG", "").rstrip("/")

# --- hermes (background delegate only, never on the voice path) ----------
HERMES_DELEGATE = _env_bool("BUD_E_DELEGATE", "1")
HERMES_SESSION = os.environ.get("BUD_E_SESSION", "bud-e-background")
HERMES_TOOLSETS = os.environ.get("BUD_E_TOOLSETS", "file,web,skills,memory")
VOICE_SKILL = "bud-e-voice"
HERMES_TIMEOUT_S = int(os.environ.get("BUD_E_DELEGATE_TIMEOUT", "900"))

# --- audio -------------------------------------------------------------
SR = 16_000
FRAME = 512                  # Silero requires exactly 512 samples at 16 kHz
MIN_SPEECH_S = 1.2           # below this, speaker verification is a coin flip
MAX_SPEECH_S = 20.0
FOLLOWUP_WINDOW_S = 30       # wake phrase not needed inside this window
DEFAULT_THRESHOLD = 0.35     # placeholder only; calibrate overwrites it

VAD_THRESHOLD = float(os.environ.get("BUD_E_VAD", "0.5"))
VAD_SILENCE_MS = int(os.environ.get("BUD_E_SILENCE_MS", "450"))
BARGE_RMS = float(os.environ.get("BUD_E_BARGE_RMS", "0.03"))

# While Bud-E is speaking, its own voice reaches the mic. Interrupting then
# needs real loudness; raise it if Bud-E cuts itself off, or wear headphones.
BARGE_RMS_SPEAKING = float(os.environ.get("BUD_E_BARGE_RMS_SPEAKING", "0.08"))

# kokoro: neural, natural, local (pip install mlx-audio). say: macOS fallback.
TTS_ENGINE = os.environ.get("BUD_E_TTS", "kokoro")
KOKORO_REPO = os.environ.get("BUD_E_KOKORO", "mlx-community/Kokoro-82M-bf16")
KOKORO_VOICE = os.environ.get("BUD_E_KOKORO_VOICE", "bm_george")  # British male
KOKORO_LANG = os.environ.get("BUD_E_KOKORO_LANG", "b")            # b = British
KOKORO_SPEED = float(os.environ.get("BUD_E_KOKORO_SPEED", "1.05"))
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
TURN_LOG = HOME / "turns.jsonl"
VOCAB_PATH = HOME / "vocab.txt"
MEMORY_PATH = HOME / "memory.md"
NOTES_PATH = HOME / "notes.md"
INBOX_PATH = HOME / "inbox.md"       # finished background (Hermes) jobs

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def ensure_dirs() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
