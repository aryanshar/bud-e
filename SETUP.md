# Setup, start to finish

Everything below is a one-time setup except the last section.

## 1. Python

```bash
python3 -m venv jarvis          # or reuse an existing venv
source jarvis/bin/activate
uv pip install -r requirements.txt          # bench only, ~20MB
uv pip install -r requirements-voice.txt    # torch, speechbrain, whisper
```

## 2. Serve the models with oMLX

Use the oMLX menu bar app; there is no `omlx serve` to run. With
`auto_start_on_launch` on (the default), the server starts when the app opens,
on the port in `~/.omlx/settings.json` (8250 here). Bud-E reads that port and
the API key from the same file.

- Add oMLX to System Settings → General → Login Items so it is always up.
- Cache settings already live in the app: SSD cache on, hot cache 6GB.
- Optional, resets on reboot: `sudo sysctl iogpu.wired_limit_mb=40960`.
  Without it Apple's default cap (~37GB) applies, enough for Qwen and Gemma.
- Gemma lives in LM Studio's folder; it is symlinked into `~/.omlx/models`.
  oMLX scans models only at startup, so restart the app after adding one.

In the admin panel, for the Qwen 27B:

| setting | value |
| --- | --- |
| quant | 4-bit (or 4-bit DWQ). 6-bit only if the bench behaviour score is under 85% |
| pin | on, no TTL. An eviction costs a reload and a cold prefill |
| context | 64k (Hermes needs it for delegate jobs); 32k if you never delegate |
| chat template kwargs | `enable_thinking: false` as the default |
| sampling | temp 0.7, top_p 0.8, top_k 20, min_p 0, presence 1.5 |

Bud-E sends the sampling and thinking switch on every request anyway, so the
panel values only matter for other clients. Pin Gemma 4 E4B too if the fast
tier stays on.

Bud-E reads the port and API key from `~/.omlx/settings.json`, so no exports
are needed. Override with `BUD_E_BASE`, `BUD_E_API_KEY`, `BUD_E_MODEL` if you
change models.

```bash
python -m bud_e.cli doctor
```

## 3. Benchmark before you trust the model

```bash
python -m bench.cli run --model "$BUD_E_MODEL" --runs 3
python -m bench.cli report
python -m bud_e.cli latency
```

Targets: behaviour 85% or better, zero malformed tool arguments, warm TTFT
under 0.8s, decode over 15 tok/s (speech only needs about 4.5). If `latency`
says warm is barely faster than cold, the prefix cache is not hitting and
nothing else matters until it does.

## 4. Tools, in text, before the microphone

```bash
brew install ical-buddy          # calendar incl. recurring events
python -m bud_e.cli chat
```

Try: "what's on tomorrow", "remind me to call the plumber Tuesday at four",
"will it rain today", "I prefer black coffee" then "how do I take my coffee".
macOS asks once for Calendar and Reminders access for your terminal.

Optional background jobs via Hermes:

```bash
cat hermes/config.yaml           # merge into ~/.hermes/config.yaml
cp -r hermes/skills/bud-e-voice ~/.hermes/skills/
```

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

Startup prefills the agent prompt once (a second or two). Then:

- "what time is it"  → trigger, instant
- "how's it going"   → fast tier, about a second
- "remind me to call the plumber at six" → agent + set_reminder, a second or two
- "plan out my week" → agent with thinking, a few seconds after "give me a moment"

Interrupt it mid-sentence to test barge-in. Have someone else speak to confirm
the speaker gate rejects them.

## 8. Tune with data, not feel

```bash
python -m bud_e.cli profile     # per-stage timing, median and share
python -m bud_e.cli latency     # cold vs warm TTFT, decode, fast tier verdict
```

Optimise the largest share. Anything under 15% is not worth touching however
slow it feels.

## Troubleshooting

| symptom | cause |
| --- | --- |
| `APIConnectionError` | oMLX not running, or `BUD_E_BASE` wrong |
| tools never called | bench behaviour score; try lower `BUD_E_PRESENCE` |
| every turn as slow as the first | prefix cache not hitting: `bud-e latency` |
| calendar empty but you have events | install icalBuddy; grant Calendar access |
| Bud-E interrupts itself | raise `BUD_E_BARGE_RMS`, or wear headphones |
| it ignores your interruption | lower `BUD_E_BARGE_RMS` |
| cut off mid-thought | raise `BUD_E_SILENCE_MS` |
| names transcribed wrong | add them to `~/.bud-e/vocab.txt` |
