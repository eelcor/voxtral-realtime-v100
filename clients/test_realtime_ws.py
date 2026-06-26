#!/usr/bin/env python3
"""Test-client voor de vLLM /v1/realtime websocket (Voxtral-realtime op V100).

Streamt een WAV (16kHz PCM16) in 100ms-chunks en print de live transcriptie.
Meet RTF (real-time factor) en first-token-latency.

Gebruik:
  python test_realtime_ws.py /tmp/sample.wav            # zo snel mogelijk (throughput)
  python test_realtime_ws.py /tmp/sample.wav --pace     # in real-time tempo (zoals een mic)
"""
import asyncio
import base64
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import websockets

URL = os.environ.get("VOXTRAL_WS_URL", "ws://localhost:8045/v1/realtime")
MODEL = os.environ.get("VOXTRAL_MODEL", "voxtral-realtime")
CHUNK = 1600  # 100 ms @ 16kHz


async def run(wav_path, pace):
    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    assert sr == 16000, f"verwacht 16kHz, kreeg {sr}"
    dur = len(audio) / sr
    print(f"[audio] {dur:.1f}s, pace={'realtime' if pace else 'max'}", file=sys.stderr, flush=True)

    deltas = []
    t0 = None
    t_first = None
    done = asyncio.Event()

    async with websockets.connect(URL, max_size=None, ping_interval=None) as ws:

        async def receiver():
            nonlocal t_first
            async for msg in ws:
                ev = json.loads(msg)
                t = ev.get("type")
                if t == "transcription.delta":
                    if t_first is None:
                        t_first = time.time()
                    deltas.append(ev["delta"])
                    print(ev["delta"], end="", flush=True)
                elif t == "transcription.done":
                    print()
                    done.set()
                    return
                elif t == "error":
                    print(f"\n[SERVER ERROR] {ev}", file=sys.stderr)
                    done.set()
                    return

        rt = asyncio.create_task(receiver())
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))

        t0 = time.time()
        started = False
        for i in range(0, len(audio), CHUNK):
            seg = audio[i:i + CHUNK]
            b64 = base64.b64encode(seg.tobytes()).decode("ascii")
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
            if not started:
                # commit(final=False) start de generatie; queue wordt live geconsumeerd
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                started = True
            if pace:
                await asyncio.sleep(len(seg) / sr)
        # trailing stilte zodat de laatste woorden uit de delay-buffer geflusht worden
        silence = np.zeros(CHUNK * 6, dtype=np.int16)  # ~600ms
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(silence.tobytes()).decode("ascii")}))
        # signaleer einde
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        except asyncio.TimeoutError:
            print("\n[timeout wachtend op transcription.done]", file=sys.stderr)
        rt.cancel()

    wall = time.time() - t0
    ftl = (t_first - t0) if t_first else float("nan")
    print(f"\n=== RESULTAAT ===", file=sys.stderr)
    print(f"  transcriptie : {''.join(deltas)!r}", file=sys.stderr)
    print(f"  audioduur    : {dur:.1f}s", file=sys.stderr)
    print(f"  wall         : {wall:.1f}s", file=sys.stderr)
    print(f"  RTF          : {wall/dur:.2f}  ({'HOUDT BIJ' if wall < dur else 'te traag'})", file=sys.stderr)
    print(f"  first-token  : {ftl:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sample.wav"
    pace = "--pace" in sys.argv
    asyncio.run(run(wav, pace))
