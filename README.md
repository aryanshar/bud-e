# bud-e

Local-first voice agent. A benchmark harness that decides which model earns the
job, and the pipeline that runs it.

```
mic → Silero VAD → ECAPA speaker check → Whisper
                        ↓ (not you)          ↓
                     discarded            router
                                             │
        ┌──────────────────────┬─────────────┴────────────────────┐
        ↓                      ↓                                  ↓
  tier 0: triggers     tier 1: Gemma 4 E4B (optional)   tier 2: Qwen 27B agent
  regex, no model      no tools, streamed               direct to oMLX, streamed,
  ~10ms                ~0.5s to first word              native tool calls,
                            │ ROUTE ───────────────────→  thinking on demand
                                                         ~0.5–1s to first word
                                                                  │ delegate
                                                                  ↓
                                                      Hermes, background only
```

Every model tier streams sentence by sentence, so speech starts at the first
full stop. The mic runs on its own thread, so speaking over Bud-E stops it
mid-sentence and cancels generation.

## Tiers

| tier | model | prompt | when |
| --- | --- | --- | --- |
| triggers | none | none | time, date, volume, weather one-liner, thanks, stop |
| fast | `gemma-4-E4B-it-MLX-4bit` | ~200 tok | chat; off with `BUD_E_FAST_TIER=0` |
| agent | `Qwen3.8-27B-MLX-4bit` | ~1.5k tok, frozen | tools, memory, depth |
| delegate | Hermes + the same 27B | ~13k tok | long jobs, answered later |

`bud_e/router.py` sends tool words and long turns to the agent, and depth
words (prove, design, debug, plan out) to the agent with thinking on.
Everything else tries the fast tier, which emits `ROUTE` when it cannot cope.

The agent's system prompt never changes; the date and time ride at the front
of each user message. That keeps the server's prefix cache hitting on every
turn, which is most of the latency win over the old Hermes path.

### Agent tools

`remember`, `recall` (`~/.bud-e/memory.md`), `add_note`, `read_notes`,
`find_files`, `read_file` (read-only, inside `BUD_E_READ_ROOTS`),
`get_calendar` (icalBuddy, AppleScript fallback), `set_reminder`,
`list_reminders` (Reminders.app), `get_weather` (Open-Meteo, no key),
`web_search` (only if `BUD_E_SEARXNG` points at a local SearXNG), and
`delegate` (only if `hermes` is on PATH).

Test them in text before using the mic:

```bash
python -m bud_e.cli chat            # add --think for depth mode
```

## Setup

Uses your own virtual environment. uv is the installer, nothing more.

```bash
source ~/jarvis/bin/activate          # or wherever your venv lives
uv pip install -r requirements.txt    # bench only, ~20MB
```

Run everything as modules from the repo root. There is no package to install.

```bash
python -m bench.cli --help
python -m bud_e.cli --help
```

Optional shell aliases:

```bash
alias bud-bench='python -m bench.cli'
alias bud-e='python -m bud_e.cli'
```

## 1. Benchmark

Serve the model from oMLX (SETUP.md step 2), then:

```bash
python -m bench.cli models                      # what the endpoint has loaded
python -m bench.cli run --model qwen3.8-27b --runs 3
python -m bench.cli run --model qwen3.8-27b-abliterated --runs 3
python -m bench.cli report
```

Ten cases cover tool selection and plain answers, using the agent's real
system prompt and tool schemas.
Targets: median TTFT under 1.00s, mean throughput above 20 tok/s, behaviour at
85% or better, zero malformed tool arguments.

Behaviour is the number that decides. Below 85% no amount of prompt work in
Hermes will save it, and the answer is a different model rather than a bigger
quant. Re-run after any change to quant, context length, or KV cache type;
those settings interact in ways that are not obvious.

```bash
export BUD_E_MODEL=Qwen3.8-27B-MLX-4bit     # the winner
```

### Thinking

Qwen thinks by default, and on a voice path that is silence. Every request
sends `chat_template_kwargs.enable_thinking=false` except turns the router
marks as depth. The bench `thinking` line should read 0; `bud-e doctor`
checks it too. If it does not, oMLX is not passing the kwarg through: set it
in the model's chat template kwargs in the admin panel.

## 3. Voice

```bash
uv pip install -r requirements-voice.txt
python -m bud_e.cli enrol         # six clips, builds your voiceprint
python -m bud_e.cli calibrate     # you vs others, writes the threshold
python -m bud_e.cli run
```

Enrol and calibrate in the room and on the mic you will actually use. If
calibrate reports overlapping distributions, re-enrol with longer clips rather
than lowering the threshold by hand.

### First test, in this order

1. "Bud-E, what day is it."
2. Within thirty seconds, without the wake phrase: "and what about tomorrow."
3. Have someone else, or a video, say "Bud-E, what day is it."

One and two exercise the speaker gate, wake phrase, follow-up window, and
session continuity. Three should print `[ignored, speaker score 0.xx]` and stay
silent.

Expect under a second to the first word on chat and simple tool turns once
the prefix cache is warm, and a few seconds on thinking turns (Bud-E says
"give me a moment" first). `bud-e latency` tells you whether the cache is
hitting and whether the fast tier is worth its memory.

Speaker verification is a filter, not a lock. It keeps a housemate or the
television out. It does not survive a recording of you played at the mic, so
nothing destructive should sit behind it alone.

## Tuning

Every turn logs its stage timings to `~/.bud-e/turns.jsonl`.

```bash
python -m bud_e.cli profile     # median, p90 and share per stage
python -m bud_e.cli latency     # cold vs warm TTFT, decode tok/s, fast tier verdict
```

Optimise the largest share, not the slowest-feeling stage. Anything under 15%
is not worth touching.

### Persona

```bash
export BUD_E_ADDRESS=sir
export BUD_E_PERSONA="Drier than that. Never enthusiastic."
```

Wake words are `bud-e`, `buddy` and `jarvis`. Inside 30 seconds of the last
exchange no wake word is needed.

### Knobs

| var | default | effect |
| --- | --- | --- |
| `BUD_E_ASR` | `mlx-whisper` | `parakeet` is faster, English only, weaker on accents |
| `BUD_E_WHISPER` | `whisper-large-v3-turbo` | swap to `whisper-large-v3-mlx` for accuracy over speed |
| `BUD_E_SILENCE_MS` | `450` | how long a pause ends your turn; lower feels snappier, cuts you off mid-thought |
| `BUD_E_BARGE_RMS` | `0.03` | loudness needed to interrupt; raise it if Bud-E interrupts itself |
| `BUD_E_TTS` | `kokoro` | neural local voice via mlx-audio; `say` for the macOS fallback |
| `BUD_E_KOKORO_VOICE` | `bm_george` | Kokoro voice; `bm_lewis`, `bf_emma`, `af_heart` are others |
| `BUD_E_KOKORO_SPEED` | `1.05` | speaking pace |
| `BUD_E_BARGE_RMS_SPEAKING` | `0.08` | loudness to interrupt while Bud-E talks; raise if it cuts itself off |
| `BUD_E_NAME` | `Aryan` | what it calls you |
| `BUD_E_VOICE` | `Daniel` | `say` fallback voice, any from `say -v '?'` |
| `BUD_E_FAST_MODEL` | `gemma-4-E4B-it-MLX-4bit` | the conversational tier |
| `BUD_E_FAST_TIER` | `1` | `0` sends chat to the 27B too |
| `BUD_E_MAX_TOKENS` | `512` | agent reply cap, non-thinking |
| `BUD_E_THINK_MAX_TOKENS` | `4096` | agent cap on thinking turns |
| `BUD_E_HISTORY` | `12` | turns kept; trimmed in halves to protect the cache |
| `BUD_E_PRESENCE` | `1.5` | Qwen presence penalty; lower if tool args get mangled |
| `BUD_E_READ_ROOTS` | `~/Documents:~/Desktop:~/Downloads` | where file tools may read |
| `BUD_E_SEARXNG` | empty | local SearXNG URL enables `web_search` |
| `BUD_E_LAT` / `BUD_E_LON` | Bengaluru | weather location |
| `BUD_E_ADDRESS` | `sir` | how it addresses you |
| `BUD_E_RATE` | `195` | words per minute |

### Accents and proper nouns

`~/.bud-e/vocab.txt` is fed to Whisper as an initial prompt on every
transcription. Adding the words you actually say — names, tools, jargon — fixes
more recognition errors than changing model does. Edit it and restart.

### Barge-in

Bud-E stops talking the moment you start. Speech onset triggers it, not a
finished sentence, so it feels immediate. Without headphones the mic hears the
laptop speakers, so there is a loudness gate: raise `BUD_E_BARGE_RMS` if it
interrupts itself, lower it if it ignores you. Headphones remove the problem
entirely, and proper echo cancellation is the only real fix if you want
open-air interruption to be reliable.

## Layout

| path | what |
| --- | --- |
| `bud_e/config.py` | every tunable; bench and pipeline read the same values |
| `bud_e/llm.py` | local endpoint client and Qwen sampling |
| `bud_e/agent.py` | the 27B agent loop: streaming, tool calls, thinking |
| `bud_e/tools.py` | every tool the agent can call |
| `bud_e/audio.py` | capture, VAD segmentation, TTS |
| `bud_e/speaker.py` | ECAPA embedding, enrolment, calibration |
| `bud_e/asr.py` | MLX Whisper on Apple silicon, faster-whisper elsewhere |
| `bud_e/triggers.py` | phrase table that bypasses every model |
| `bud_e/router.py` | which tier takes the turn |
| `bud_e/persona.py` | how Bud-E talks; shared by both tiers, sized to fill oMLX's cache block |
| `bud_e/fast.py` | the small-model tier, streamed by sentence |
| `bud_e/ear.py` | the loop, with barge-in and cancellation |
| `bud_e/profile.py` | per-stage timing and the summary report |
| `bench/` | cases, runner, comparison report |
| `requirements*.txt` | base stack, then the audio stack |
| `results/` | one JSON per benchmarked model |
| `hermes/` | config and skill for background delegate jobs |

## Environment

| var | default |
| --- | --- |
| `BUD_E_BASE` | read from `~/.omlx/settings.json` (port 8250 here) |
| `BUD_E_API_KEY` | read from `~/.omlx/settings.json` |
| `BUD_E_MODEL` | `Qwen3.8-27B-MLX-4bit` |
| `BUD_E_HOME` | `~/.bud-e` |

## Deliberately not enabled

No shell tool in the agent, and `terminal` is absent from `HERMES_TOOLSETS`.
Voice dispatch means no human reads the command before it runs. File tools
are read-only and fenced to `BUD_E_READ_ROOTS`. Add shell access on purpose, with the Docker
mount list decided, not because a turn failed and shell would have fixed it.
