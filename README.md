# bud-e

Local-first voice agent. A benchmark harness that decides which model earns the
job, and the pipeline that runs it.

```
mic → Silero VAD → ECAPA speaker check → Whisper
                        ↓ (not you)          ↓
                     discarded            router
                                             │
        ┌────────────────────────────────────┼──────────────────────────┐
        ↓                                    ↓                          ↓
  tier 0: triggers              tier 1: Gemma 4 E4B          tier 2: Hermes + Qwen 27B
  regex, no model               direct, no tools             tools, skills, memory
  ~10ms                         ~1s, streamed                ~3-5s
                                     │ ROUTE ─────────────────────→ ┘
```

Tier 1 streams sentence by sentence, so speech starts before generation
finishes. Tier 2 is where you wait on purpose.

## Tiers

| tier | model | prompt | when |
| --- | --- | --- | --- |
| triggers | none | none | time, date, volume, thanks, stop |
| fast | `google/gemma-4-e4b` | ~200 tok | conversation, general knowledge |
| hermes | `qwen3.8-27b-uncensored-mlx` | ~13k tok | tools, memory, files, web, depth |

`bud_e/router.py` hard-routes to Hermes on tool words, depth words, or turns
over 28 words. Everything else tries the fast tier, which emits `ROUTE` when
it cannot cope. The model reports failure; it never predicts difficulty.

Load both in LM Studio, they coexist in 48GB:

```bash
lms load qwen3.8-27b-uncensored-mlx --context-length 64000
lms load google/gemma-4-e4b --context-length 8192
lms ps
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

Load a model in LM Studio at 64k context with Q8_0 KV cache, start the server,
then:

```bash
python -m bench.cli models                      # what the endpoint has loaded
python -m bench.cli run --model qwen3.8-27b --runs 3
python -m bench.cli run --model qwen3.8-27b-abliterated --runs 3
python -m bench.cli report
```

Eight cases cover tool selection, plain answers, and the ESCALATE contract.
Targets: median TTFT under 1.00s, mean throughput above 20 tok/s, behaviour at
85% or better, zero malformed tool arguments.

Behaviour is the number that decides. Below 85% no amount of prompt work in
Hermes will save it, and the answer is a different model rather than a bigger
quant. Re-run after any change to quant, context length, or KV cache type;
those settings interact in ways that are not obvious.

```bash
export BUD_E_MODEL=qwen3.8-27b-uncensored     # the winner
```

### Thinking

Qwen3.x reasons before answering, and on a voice path that is pure latency:
400 reasoning tokens at 17 tok/s is 23 seconds of silence. The soft switch is
parsed from the latest **user** message, not the system prompt, so `/no_think`
is appended to the query in both the bench and `bud_e/llm.py`. The bench
`thinking` line should read 0. If it does not, check LM Studio's reasoning
toggle in the model's gear panel before doing more prompt archaeology.

Keep `max_tokens` generous. At 400 the model spends the whole budget reasoning
and emits no content at all, which shows up as an empty reply rather than as
an error.

## 2. Wire Hermes

```bash
hermes model                       # → LM Studio → localhost:1234/v1
cat hermes/config.yaml             # merge into ~/.hermes/config.yaml
cp -r hermes/skills/bud-e-voice ~/.hermes/skills/
hermes prompt-size                 # want under ~8k
python -m bud_e.cli doctor
```

Prove it in text before touching the microphone:

```bash
hermes chat --oneshot -Q -c bud-e -t file,web,skills,memory -s bud-e-voice \
  -q "what day is it"
```

Two spoken sentences. Then ask it to prove something hard; it should return
`ESCALATE:` and nothing else. If it attempts the proof, the skill is not
loading.

## Local only

No API keys. There is no cloud tier unless you explicitly turn one on. When the
27B declines a turn, Bud-E says so and appends the request to
`~/.bud-e/escalations.log`. After a week that file answers the question
empirically: if it is nearly empty, you never needed a second tier; if it fills
up with the same kind of request, you know exactly what you would be paying for.

To turn escalation on later:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> ~/.hermes/.env
export BUD_E_CLOUD=1
```

Both are required. A key alone does nothing.

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

Expect two to four seconds per reply. That is where a 27B lands on this
hardware. Bud-E speaks a short acknowledgement while the model works, and the
mic is drained afterwards so it does not hear its own voice. The two levers
that actually reduce latency are the trigger table, which returns in
milliseconds, and moving off `hermes chat --oneshot` to `hermes serve` to skip
per-turn process startup. Do neither until you have lived with it for a few
days and the escalation log tells you which turns are worth the effort.

Speaker verification is a filter, not a lock. It keeps a housemate or the
television out. It does not survive a recording of you played at the mic, so
nothing destructive should sit behind it alone.

## Tuning

Every turn logs its stage timings to `~/.bud-e/turns.jsonl`.

```bash
python -m bud_e.cli profile     # median, p90 and share per stage
python -m bud_e.cli latency     # splits LLM time: spawn vs model vs hermes
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

### Cold start

The Hermes system prompt is ~13k tokens and prefill runs at roughly 400
words/sec, so the first turn after an LM Studio restart costs about 25
seconds. `bud-e run` pays that at startup deliberately. Subsequent turns reuse
the KV cache. If a turn is unexpectedly slow mid-session, the session history
has grown past the cached prefix; lower `compression.threshold`.

### Knobs

| var | default | effect |
| --- | --- | --- |
| `BUD_E_ASR` | `mlx-whisper` | `parakeet` is faster, English only, weaker on accents |
| `BUD_E_WHISPER` | `whisper-large-v3-turbo` | swap to `whisper-large-v3-mlx` for accuracy over speed |
| `BUD_E_SILENCE_MS` | `380` | how long a pause ends your turn; lower feels snappier, cuts you off mid-thought |
| `BUD_E_BARGE_RMS` | `0.03` | loudness needed to interrupt; raise it if Bud-E interrupts itself |
| `BUD_E_VOICE` | `Daniel` | any voice from `say -v '?'` |
| `BUD_E_FAST_MODEL` | `google/gemma-4-e4b` | the conversational tier |
| `BUD_E_REASONING` | `low` | Hermes `--reasoning`; empty string to omit |
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
| `bud_e/llm.py` | local endpoint client, Hermes wrapper, ESCALATE parsing |
| `bud_e/audio.py` | capture, VAD segmentation, TTS |
| `bud_e/speaker.py` | ECAPA embedding, enrolment, calibration |
| `bud_e/asr.py` | MLX Whisper on Apple silicon, faster-whisper elsewhere |
| `bud_e/triggers.py` | phrase table that bypasses every model |
| `bud_e/router.py` | which tier takes the turn |
| `bud_e/fast.py` | the small-model tier, streamed by sentence |
| `bud_e/ear.py` | the loop, with barge-in and cancellation |
| `bud_e/profile.py` | per-stage timing and the summary report |
| `bench/` | cases, runner, comparison report |
| `requirements*.txt` | base stack, then the audio stack |
| `results/` | one JSON per benchmarked model |
| `hermes/` | config fragment and the voice skill |

## Environment

| var | default |
| --- | --- |
| `BUD_E_BASE` | `http://localhost:1234/v1` |
| `BUD_E_MODEL` | `qwen3.8-27b` |
| `BUD_E_CLOUD` | `0` (local only; `1` enables escalation) |
| `BUD_E_CLOUD_MODEL` | `claude-sonnet-4-6` |
| `BUD_E_HOME` | `~/.bud-e` |

## Deliberately not enabled

`terminal` is absent from `HERMES_TOOLSETS`. Voice dispatch means no human
reads the command before it runs. Add shell access on purpose, with the Docker
mount list decided, not because a turn failed and shell would have fixed it.
