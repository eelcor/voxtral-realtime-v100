# Voxtral Realtime — web app

A polished browser app for live speech-to-text: record on/off, streaming transcription,
a voice-band volume meter, and a live waveform — all talking to your local Voxtral realtime
server.

![record · transcribe · visualize](https://img.shields.io/badge/record-transcribe-6366f1) ·
runs against the `serve_voxtral_vllm.sh` server from the parent repo.

## What it does

- 🎙️ **Record on/off** — big button + <kbd>space</kbd> shortcut
- 📝 **Live transcription** — text streams in word-by-word as you speak
- 🎚️ **Voice-band volume meter** — a glow ring + level bar weighted to 200–3800 Hz, so it
  reacts to *speech*, not rumble; plus a clip indicator
- 📊 **Live waveform** — animated frequency bars
- ⏱️ Recording timer · word/char count
- 🎙️ Microphone selector · 🔌 model-status indicator · 🌓 dark/light theme
- 📋 Copy · 💾 Download `.txt` · 🗑️ Clear
- 🌐 Auto language detection (Dutch + 12 more)
- 🔒 Audio never leaves your network (browser → your server → your Voxtral)

## Signal processing

Beyond the browser's built-in `echoCancellation` / `noiseSuppression` / `autoGainControl`,
two toggles add a deliberate Web-Audio chain before the audio is sent to Voxtral:

- **Ruisfilter (enhance)** — high-pass ~80 Hz (kills rumble/handling/DC), 50 Hz notch
  (mains hum), anti-alias low-pass ~7.5 kHz (before the 16 kHz downsample), and a mild
  compressor (evens out loud/soft speech).
- **Stiltepoort (noise gate / VAD)** — sends digital silence during pauses (adaptive noise
  floor + 300 ms hold), so background noise isn't transcribed. Timing stays continuous
  (silence frames, not dropped frames) to keep the streaming model in sync.

The volume meter itself is voice-band weighted, and a clip indicator warns on overload.

## Run

```bash
# 1) Have a Voxtral realtime server running (see parent repo)
VLLM_VENV=$HOME/vllm-v100 ../serve_voxtral_vllm.sh        # serves ws://localhost:8045

# 2) Start the web app (generates a self-signed cert on first run, serves HTTPS)
VLLM_VENV=$HOME/vllm-v100 ./run_webapp.sh                  # https://0.0.0.0:8443
```

Then open **`https://<server-ip>:8443`** and accept the self-signed certificate warning.

Env: `WEBAPP_PORT` (8443), `VOXTRAL_WS` (`ws://127.0.0.1:8045/v1/realtime`),
`VLLM_VENV`, `VOXTRAL_MODEL` (`voxtral-realtime`).

### Why HTTPS?

Browsers only expose the microphone in a **secure context**. So the app is served over
HTTPS with a self-signed cert (click *Advanced → proceed*). Alternatively, tunnel it and use
`localhost` (also a secure context): `ssh -L 8443:localhost:8443 user@server`, then open
`https://localhost:8443`.

## Architecture

```
browser  ──wss──►  run_webapp.sh (FastAPI)  ──ws──►  vLLM /v1/realtime  ──►  Voxtral
  mic → 16 kHz PCM16          /ws proxy              input_audio_buffer        transcription
  + DSP + meters                                     .append / .commit
```

The browser captures the mic, downsamples to 16 kHz mono PCM16 in an `AudioWorklet`, applies
the DSP chain, and streams raw frames to the backend. The backend relays them to the vLLM
realtime websocket and streams `transcription.delta` events back. One port; the vLLM port is
never exposed to the browser.

## Files

```
server.py                 FastAPI: serves the app + /ws proxy to vLLM realtime
run_webapp.sh             generates the self-signed cert and launches uvicorn (HTTPS)
test_proxy.py             headless end-to-end check (streams a WAV through /ws)
static/index.html         the UI (dark, gradient, glassmorphism)
static/app.js             mic capture, DSP chain, meters, websocket client
static/recorder-worklet.js  48 kHz → 16 kHz PCM16 downsampler (AudioWorklet)
```
