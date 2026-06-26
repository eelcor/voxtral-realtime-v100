#!/usr/bin/env python3
"""Live realtime-transcriptie: microfoon -> vLLM /v1/realtime (Voxtral op V100).

Vangt de microfoon via pw-record (PipeWire) of arecord (ALSA) en streamt de
audio naar de vLLM-websocket. Transcriptie verschijnt live. Houdt real-time bij
(gemeten RTF ~0.4 op de V100), i.t.t. het transformers-pad.

Gebruik:
  python mic_realtime_vllm.py                              # PipeWire default-bron
  python mic_realtime_vllm.py --source <pipewire-node>     # andere bron (wpctl status)
  python mic_realtime_vllm.py --backend arecord --source plughw:CARD=PCH,DEV=0

Env: VOXTRAL_WS_URL, VOXTRAL_MODEL, VOXTRAL_PW_SOURCE.
Stoppen: Ctrl+C.
"""
import argparse
import asyncio
import base64
import json
import os
import signal
import sys

import websockets

URL = os.environ.get("VOXTRAL_WS_URL", "ws://localhost:8045/v1/realtime")
MODEL = os.environ.get("VOXTRAL_MODEL", "voxtral-realtime")
SR = 16000
CHUNK_BYTES = 1600 * 2  # 100 ms PCM16
# None => PipeWire default-bron. Zet VOXTRAL_PW_SOURCE of geef --source.
DEFAULT_PW_SOURCE = os.environ.get("VOXTRAL_PW_SOURCE")


def capture_cmd(backend, source):
    if backend == "pipewire":
        tgt = source or DEFAULT_PW_SOURCE
        cmd = ["pw-record", "--rate", str(SR), "--channels", "1", "--format", "s16"]
        if tgt:
            cmd += ["--target", tgt]
        return cmd + ["-"]
    dev = source or "default"
    return ["arecord", "-D", dev, "-f", "S16_LE", "-r", str(SR),
            "-c", "1", "-t", "raw", "-q", "-"]


async def main(backend, source):
    cmd = capture_cmd(backend, source)
    print(f"[bron] {' '.join(cmd)}", file=sys.stderr, flush=True)
    rec = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    stop = asyncio.Event()

    async with websockets.connect(URL, max_size=None, ping_interval=None) as ws:

        async def receiver():
            async for msg in ws:
                ev = json.loads(msg)
                t = ev.get("type")
                if t == "transcription.delta":
                    print(ev["delta"], end="", flush=True)
                elif t == "transcription.done":
                    print()
                elif t == "error":
                    print(f"\n[SERVER ERROR] {ev}", file=sys.stderr, flush=True)

        async def sender():
            await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
            started = False
            while not stop.is_set():
                data = await rec.stdout.readexactly(CHUNK_BYTES)
                if not data:
                    break
                b64 = base64.b64encode(data).decode("ascii")
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                if not started:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    started = True
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        print("[klaar] Praat maar — transcriptie verschijnt live. Ctrl+C om te stoppen.\n",
              file=sys.stderr, flush=True)
        rt = asyncio.create_task(receiver())
        st = asyncio.create_task(sender())
        await stop.wait()
        st.cancel()
        # geef de receiver even om de laatste deltas te tonen
        try:
            await asyncio.wait_for(rt, timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    try:
        rec.terminate()
    except Exception:
        pass
    print("\n[gestopt]", file=sys.stderr, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["pipewire", "arecord"], default="pipewire")
    p.add_argument("--source", default=None)
    args = p.parse_args()
    try:
        asyncio.run(main(args.backend, args.source))
    except KeyboardInterrupt:
        pass
