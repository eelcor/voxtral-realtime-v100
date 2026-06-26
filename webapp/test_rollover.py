#!/usr/bin/env python3
"""Test de sessie-rollover: streamt een lang fragment door /ws (real-time tempo),
accumuleert over meerdere segmenten (rollovers), en checkt dat de server overleeft."""
import asyncio, json, ssl, sys, time
import numpy as np, soundfile as sf, websockets

URL = "wss://127.0.0.1:8443/ws"
WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/long.wav"
CHUNK = 1600

async def main():
    audio, sr = sf.read(WAV, dtype="int16")
    if audio.ndim > 1: audio = audio[:, 0]
    assert sr == 16000
    dur = len(audio) / sr
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    committed, live, segments, errors = [], "", 0, []
    async with websockets.connect(URL, ssl=ctx, max_size=None, ping_interval=None) as ws:
        async def recv():
            nonlocal live, segments
            async for msg in ws:
                m = json.loads(msg); t = m.get("type")
                if t == "delta": live += m["text"]
                elif t == "segment_done":
                    seg = (m.get("text") or live).strip()
                    if seg: committed.append(seg)
                    live = ""; segments += 1
                elif t == "error": errors.append(m.get("msg",""))
        rt = asyncio.create_task(recv())
        await ws.send(json.dumps({"type":"start"})); await asyncio.sleep(0.3)
        t0 = time.time()
        for i in range(0, len(audio), CHUNK):
            await ws.send(audio[i:i+CHUNK].tobytes())
            await asyncio.sleep(CHUNK/sr)   # real-time
        await ws.send(json.dumps({"type":"stop"}))
        await asyncio.sleep(2.0); rt.cancel()
    full = " ".join(committed + ([live.strip()] if live.strip() else []))
    print(f"audio        : {dur:.0f}s")
    print(f"segmenten    : {segments}  (rollovers = {max(0,segments-1)})")
    print(f"fouten       : {errors if errors else 'geen'}")
    print(f"transcript   : {len(full)} tekens")
    print(f"  -> {full[:300]}{'...' if len(full)>300 else ''}")

asyncio.run(main())
