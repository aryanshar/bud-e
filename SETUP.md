# Setup, start to finish

Everything below is a one-time setup except the last section.

## 1. Python

```bash
python3 -m venv jarvis          # or reuse an existing venv
source jarvis/bin/activate
uv pip install -r requirements.txt          # bench only, ~20MB
uv pip install -r requirements-voice.txt    # torch, speechbrain, whisper
```

## 2. Models in LM Studio

Load both. Set context explicitly; auto-detect can come back under 64k and
Hermes will refuse the session at startup.

```bash
lms load qwen3.8-27b-uncensored-mlx --context-length 64000
lms load google/gemma-4-e4b --context-length 8192
lms server start
lms ps
```

In each model's gear panel: Flash Attention on. For GGUF builds, set K and V
cache quantisation to Q8_0 (the option only appears once Flash Attention is
enabled). Save as per-model defaults so a reload does not drop them.

## 3. Benchmark before you trust either model

```bash
python -m bench.cli models
python -m bench.cli run --model qwen3.8-27b-uncensored-mlx --runs 3
python -m bench.cli run --model google/gemma-4-e4b --runs 3
python -m bench.cli report
```

Targets: reply under 1.50s, throughput over 20 tok/s, behaviour at 85% or
better, zero malformed tool arguments. Behaviour is the number that decides.
Below 85% no prompt work will save it and the answer is a different model.

If the `thinking` line is not 0, reasoning is still on and that is where your
latency is going. Check LM Studio's reasoning toggle in the gear panel.

## 4. Hermes

```bash
hermes model
```

Custom OpenAI-compatible endpoint → `http://localhost:1234/v1` → blank key →
API mode **2, Chat Completions** → pick the Qwen → context **64000** → display
name `Bud-E`.

Verify, because this is the step that silently leaves you on a remote model:

```bash
hermes config get model.provider     # custom
hermes config get model.default      # your qwen
hermes chat                          # banner bottom-left names your model
```

Then the config fragment and the skill:

```bash
cat hermes/config.yaml               # merge into ~/.hermes/config.yaml
cp -r hermes/skills/bud-e-voice ~/.hermes/skills/
hermes skills list | grep bud-e-voice
```

Prove it in text before touching the microphone:

```bash
hermes chat --oneshot -Q -c bud-e-voice --create-if-missing \
  -t file,skills,memory -s bud-e-voice -q "what day is it /no_think"
```

Two spoken sentences, no markdown. If you get an essay with headings, the
skill is not being applied and the voice path will inherit that.

## 5. Voice enrolment

macOS asks for microphone access on first run. If no prompt appears, grant it
under Privacy & Security → Microphone and restart the terminal. A denied mic
returns silence rather than an error, and enrolment will happily build a
voiceprint out of nothing.

```bash
python -m bud_e.cli enrol
python -m bud_e.cli calibrate
```

Do both in the room and on the mic you actually use. If calibrate reports
overlapping distributions, re-enrol with longer clips rather than lowering the
threshold by hand.

## 6. Check the wiring

```bash
python selfcheck.py
```

Verifies every module imports and every config reference resolves. Run it after
any edit; it catches missing config attributes that otherwise only appear when
you speak to it.

## 7. Run

```bash
./run.sh
```

Startup pays the ~25s Hermes prefill once. Then:

- "what time is it"  → trigger, instant
- "how's it going"   → fast tier, about a second
- "remind me to call the plumber" → Hermes, three to five seconds

Interrupt it mid-sentence to test barge-in. Have someone else speak to confirm
the speaker gate rejects them.

## 8. Tune with data, not feel

```bash
python -m bud_e.cli profile     # per-stage timing, median and share
python -m bud_e.cli latency     # splits LLM time: spawn vs model vs hermes
```

Optimise the largest share. Anything under 15% is not worth touching however
slow it feels.

## Troubleshooting

| symptom | cause |
| --- | --- |
| `APIConnectionError` | LM Studio server not started: `lms server start` |
| "No session found matching" | missing `--create-if-missing` |
| replies are markdown essays | skill not applied; check `hermes skills list` |
| first turn ~25s, rest fast | normal cold prefill, warm-up handles it |
| every turn slow | check `hermes chat` banner names your local model |
| Bud-E interrupts itself | raise `BUD_E_BARGE_RMS`, or wear headphones |
| it ignores your interruption | lower `BUD_E_BARGE_RMS` |
| cut off mid-thought | raise `BUD_E_SILENCE_MS` |
| names transcribed wrong | add them to `~/.bud-e/vocab.txt` |
