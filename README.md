# Bud-E

A voice assistant that runs entirely on your own machine. You talk, it
answers out loud, and it can check your calendar, set reminders, remember
things about you, read your files and look up the weather. No cloud and no
API keys: speech recognition, the language model and the voice all run
locally.

It runs on an Apple silicon Mac or on a Linux box with an NVIDIA GPU. You
can also split it: the model on a GPU machine, the microphone on your
laptop.

```
 mic ─▶ voice detection ─▶ (is it you?) ─▶ Whisper ─▶ wake word?
        Silero VAD          optional         speech       │
                                             to text      ▼
                                              instant answer (stop, time, date)
                                                          │ otherwise
                                                          ▼
        speaker ◀── Kokoro voice ◀── sentences ◀── local LLM ⇄ tools
                    (sentence 1 plays while 2 is being generated)
```

**How it works**

- **One model does everything.** Bud-E talks to a model server you run
  yourself (Ollama, LM Studio, oMLX, llama.cpp, vLLM and others) over the
  standard OpenAI chat API. The model chats, decides when to use a tool, and
  answers. There's no router and no second model.
- **It streams.** Speech starts at the first full stop, not when the whole
  reply is done.
- **You can interrupt it.** Start talking and it stops mid-sentence and drops
  what it was generating.
- **The prompt stays fixed.** The system prompt doesn't change between turns,
  so the server can reuse its cache and only processes your new words. The
  current time goes in front of each of your messages instead.

---

## 1. Pick a model for your hardware

The model needs memory (Mac unified memory, or GPU VRAM on Linux). Leave
about 3 GB free for Whisper and the voice.

| Memory | What to run | Notes |
| --- | --- | --- |
| 16 GB | a 4–8B model at 4-bit, e.g. *Qwen3-4B-Instruct-2507*, *Qwen3-8B* | Quick, but makes more tool-call mistakes |
| 24–36 GB | a ~30B mixture-of-experts model with ~3B active, at 4-bit, e.g. *Qwen3-30B-A3B-Instruct-2507* | **Best default.** Quality close to a 30B model at the speed of a 3B one |
| 48 GB+ | the same MoE at 8-bit, or a dense 27–32B at 4-bit | Dense models are noticeably slower to start talking |

The names above are examples; newer releases in the same size class should
work too. Things to look for:

- **Good tool calling.** This matters more than anything else. Run
  `bud-e bench` (step 6) and keep the model if it scores 85% or more.
- **An "instruct" model, not a "thinking" one.** Thinking models stay silent
  while they reason. Bud-E asks Qwen-style models to skip thinking, and hides
  any `<think>` text that comes through, but the time spent thinking still
  delays the reply.
- **MoE over dense if you're unsure.** Speed to first word is what makes a
  voice assistant feel good.

## 2. Run a model server

Pick one. Bud-E only needs the server's URL and the model name.

### macOS (Apple silicon)

MLX-based servers are the fastest on a Mac.

**LM Studio** (GUI, easiest): install from lmstudio.ai, download an MLX build
of your model, then go to Developer → Start Server. URL:
`http://localhost:1234/v1`.

**oMLX** (menu-bar app with an on-disk prompt cache): add the model in its
admin panel, pin it so it stays loaded, and turn off thinking in its chat
template settings. The port is in `~/.omlx/settings.json`. Bud-E reads the
API key from that file automatically.

**mlx-lm** (no GUI):

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080
# URL: http://localhost:8080/v1
```

**Ollama** (simplest to install, slower on a Mac than MLX):

```bash
brew install ollama && ollama serve        # leave this running
ollama pull qwen3:8b                       # or a bigger model that fits
# URL: http://localhost:11434/v1
```

Optional: macOS limits how much memory the GPU can use to about 75% of RAM.
To raise it until the next reboot (for example 40 GB on a 48 GB Mac):
`sudo sysctl iogpu.wired_limit_mb=40960`.

### Linux with an NVIDIA GPU

Check the driver first: `nvidia-smi` should list your GPU.

**vLLM** (fastest when the model fits in VRAM):

```bash
uv tool install vllm
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --port 8000
# URL: http://localhost:8000/v1
```

The two `--...tool...` flags are required, or tool calls come back as plain
text. FP8 needs an RTX 40-series or newer card. On a 3090 or older, use an
AWQ or GPTQ 4-bit build of the model.

**llama.cpp** (best when the model is a little too big for VRAM, since it can
keep some layers in system RAM):

```bash
llama-server -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M \
  --jinja -ngl 99 -c 32768 --port 8080
# URL: http://localhost:8080/v1
```

`--jinja` turns on tool calling. On a 12–16 GB card, add `--n-cpu-moe 20`
(tune the number) to keep part of the model in RAM.

**Ollama**: `curl -fsSL https://ollama.com/install.sh | sh`, then
`ollama pull <model>`. URL: `http://localhost:11434/v1`.

### Split setup: model on one machine, mic on another

Run the server on the GPU machine, listening on the network (vLLM
`--host 0.0.0.0`, llama.cpp `--host 0.0.0.0`, Ollama `OLLAMA_HOST=0.0.0.0`),
and set `base_url = "http://<gpu-box-ip>:<port>/v1"` on the laptop. Only
text goes over the network; your audio stays on the laptop.

## 3. Install Bud-E

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:aryanshar/bud-e.git && cd bud-e
# or: git clone https://github.com/aryanshar/bud-e.git
uv sync --extra voice                    # add --extra speaker for the voice check
```

`uv sync` installs exactly the extras you name and removes the rest. If you
use the voice check, include `--extra speaker` every time you sync.

**macOS** needs nothing else. The first time you run it, macOS asks to let
your terminal use the microphone, and later Calendar and Reminders. For
calendar events that repeat, also run `brew install ical-buddy`.

**Linux** needs a couple of system packages:

```bash
sudo apt install libportaudio2 espeak-ng   # audio I/O, and a fallback for Kokoro
```

Whisper uses the GPU on its own if CUDA works. If it fails with a missing
`libcudnn` or `libcublas`, see Troubleshooting.

Run everything through `uv run bud-e ...`, or activate the venv with
`source .venv/bin/activate` and use `bud-e ...` directly.

## 4. Configure

```bash
uv run bud-e init          # writes ~/.bud-e/config.toml
```

Open the file and set at least:

```toml
[llm]
base_url = "http://localhost:1234/v1"   # your server from step 2
model    = "qwen3-30b-a3b-instruct-2507" # as the server names it

[user]
name      = "Sam"
location  = "Bengaluru"
latitude  = 12.9716                      # these two turn on weather
longitude = 77.5946
```

The commented example
[bud_e/config.example.toml](bud_e/config.example.toml) lists every setting.
To try another server or model for one run without editing the file, set
`BUD_E_BASE_URL` / `BUD_E_MODEL`.

## 5. Check it, then try it in text

```bash
uv run bud-e doctor        # server, model, tool calls, audio devices
uv run bud-e chat          # type to it: no mic needed
```

`doctor` sends one real request and confirms the model makes tool calls. In
`chat`, try "what's on tomorrow", "remind me to call the plumber Tuesday at
four", "will it rain today", or "I take my coffee black" and later "how do I
take my coffee". Fix anything that's off here, before adding audio to the
mix.

## 6. Benchmark the model (worth doing once)

```bash
uv run bud-e bench         # about a minute
uv run bud-e bench --report  # compare every model you've tried
```

This runs ten realistic requests with the real prompt and tools. Keep a
model that picks the right tool 85% or more of the time, never writes broken
tool arguments, and starts answering in under about 1.5 s. If it scores
low, try a different model rather than a bigger quantisation of the same one.

## 7. Talk to it

```bash
uv run bud-e run
```

Say **"Bud-E, what's the weather like?"**. For 30 seconds after it answers you
don't need the wake word, so you can just carry on talking. The wake words
are `bud-e`, `buddy` and `jarvis`, and you can change them in the config.

Use **headphones** if you can. Without them the mic hears the speakers.
Bud-E ignores quiet sounds while it's talking, but headphones remove the
problem completely.

### Only answer your voice (optional)

```bash
uv sync --extra voice --extra speaker
uv run bud-e enrol         # six short clips of you
uv run bud-e calibrate     # you vs. someone else (a podcast works)
```

Once you've enrolled, it's on automatically. Do both steps in the room and on
the mic you normally use. This keeps a housemate or the TV out. It is not
security: a recording of your voice gets through.

---

## Tools

| Tool | What it does | Where |
| --- | --- | --- |
| `remember`, `recall` | Long-term memory in `~/.bud-e/memory.md`. Recent entries are also added to the prompt | everywhere |
| `add_note`, `read_notes` | `~/.bud-e/notes.md` | everywhere |
| `find_files`, `read_file` | Read-only, only inside `read_roots` | everywhere |
| `get_weather` | Open-Meteo, no key needed | when latitude/longitude are set |
| `web_search` | Your own SearXNG instance | when `searxng_url` is set |
| `set_volume` | System volume | macOS; Linux with `pactl` |
| `get_calendar`, `set_reminder`, `list_reminders` | Calendar.app and Reminders.app | macOS |

There's no shell tool on purpose. With voice, nobody reads a command before
it runs.

To add a tool, write a function in [bud_e/tools.py](bud_e/tools.py) with the
`@tool(...)` decorator. It takes the config plus its arguments and returns a
short string. `available=` controls which machines it's offered on.

## Tuning

Every voice turn is timed and logged to `~/.bud-e/turns.jsonl`:

```bash
uv run bud-e stats
```

`first_audio` is the number to watch: the time from when you stop talking to
when it starts. Work on whichever stage is biggest.

| Symptom | Fix |
| --- | --- |
| Slow to start answering | A smaller or MoE model. Check that your server has prefix caching on (vLLM and llama.cpp do by default) |
| Cuts you off mid-thought | Raise `voice.silence_ms` (try 700) |
| Interrupts itself | Raise `voice.barge_rms_speaking`, or wear headphones |
| Ignores your interruptions | Lower `voice.barge_rms_speaking` |
| Gets names and jargon wrong | Add them to `~/.bud-e/vocab.txt` (it's fed to Whisper) |
| Whisper slow on a CPU-only machine | `voice.whisper_model = "small.en"` |
| Voice sounds wrong | `voice.kokoro_voice`: `af_heart`, `bf_emma`, `bm_lewis`, `am_michael`... |

## Troubleshooting

**`doctor` says the server isn't reachable.** Start the server (step 2) and
check that the port in `base_url` matches it. `curl <base_url>/models` should
return JSON.

**Tool calls show up as text, or never happen.** The server isn't parsing
tool calls. On vLLM add `--enable-auto-tool-choice --tool-call-parser hermes`.
On llama.cpp add `--jinja`. In LM Studio, use a model it marks as
supporting tools. If that's all set and the bench score is still low, try a
different model.

**The server rejects requests (400 errors).** Bud-E retries once without
`extra_body` and prints a note when it does. To stop sending it at all, set
`extra_body = {}` under `[llm]`.

**It answers after a long silence.** It's probably a thinking model. Use the
instruct version, or turn thinking off in the server.

**No sound in, or silence recorded (macOS).** Allow your terminal under System
Settings → Privacy & Security → Microphone, then restart the terminal. When
access is denied, macOS gives the app silence rather than an error.

**Wrong mic or speaker.** Run `python -m sounddevice` to list devices, then set
`voice.input_device` / `voice.output_device` to a name or number.

**Linux: `libcudnn` or `libcublas` not found (Whisper).** Point faster-whisper
at the CUDA libraries that came with PyTorch:

```bash
export LD_LIBRARY_PATH=$(uv run python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

**Startup hangs after loading, or Kokoro won't load.** Kokoro needs spaCy's
small English model (`en-core-web-sm`), which the voice extra installs. If
the model is missing, Kokoro tries to download it on every start and can
hang there. Re-run `uv sync --extra voice` (plus any other extras you use).
If Kokoro still won't load, Bud-E falls back to the macOS `say` voice. On
Linux it prints replies instead of speaking.

## Project layout

| File | What's in it |
| --- | --- |
| [bud_e/config.py](bud_e/config.py) | Every setting, with defaults. Loads `~/.bud-e/config.toml` |
| [bud_e/llm.py](bud_e/llm.py) | The OpenAI-compatible client: streaming, tool-call assembly |
| [bud_e/agent.py](bud_e/agent.py) | The conversation loop: stream, run tools, repeat; history |
| [bud_e/prompt.py](bud_e/prompt.py) | Personality and tool rules. Built once per session |
| [bud_e/tools.py](bud_e/tools.py) | Every tool, and which machines it runs on |
| [bud_e/text.py](bud_e/text.py) | Sentence splitting, hiding `<think>`, cleaning text for speech |
| [bud_e/commands.py](bud_e/commands.py) | Instant answers that skip the model (stop, time, date) |
| [bud_e/audio.py](bud_e/audio.py) | Mic and voice detection; the interruptible speech player |
| [bud_e/asr.py](bud_e/asr.py) | Whisper: MLX on Mac, faster-whisper (CUDA or CPU) elsewhere |
| [bud_e/tts.py](bud_e/tts.py) | Kokoro, or the macOS `say` voice |
| [bud_e/speaker_id.py](bud_e/speaker_id.py) | Optional voice check: enrol, calibrate |
| [bud_e/assistant.py](bud_e/assistant.py) | The voice loop that connects all of the above |
| [bud_e/bench.py](bud_e/bench.py) | Model benchmark |
| [bud_e/turnlog.py](bud_e/turnlog.py) | Per-turn timing and `bud-e stats` |
| [tests/](tests/) | `uv run pytest`. Runs without a server, mic or GPU |

Everything Bud-E saves lives in `~/.bud-e/` (or `$BUD_E_HOME`): config,
memory, notes, vocabulary, voiceprint, turn log and bench results.

## License

Apache 2.0. See [LICENSE](LICENSE).
