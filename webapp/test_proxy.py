#!/usr/bin/env python3
"""Valideert de webapp /ws-proxy end-to-end: stuurt PCM-frames zoals de browser
zou doen, en checkt of er transcriptie terugkomt. Geen browser nodig."""
import asyncio, json, ssl, sys
import numpy as np, soundfile as sf, websockets

URL = "wss://127.0.0.1:8443/ws"
WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sample.wav"
CHUNK = 1600  # 100 ms @16k

async def main():
    audio, sr = sf.read(WAV, dtype="int16")
    if audio.ndim > 1: audio = audio[:, 0]
    assert sr == 16000, sr
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    got = []
    async with websockets.connect(URL, ssl=ctx, max_size=None, ping_interval=None) as ws:
        done = asyncio.Event()
        async def recv():
            async for msg in ws:
                m = json.loads(msg)
                if m["type"] == "delta": got.append(m["text"]); print(m["text"], end="", flush=True)
                elif m["type"] == "segment_done": print(); done.set(); return
                elif m["type"] == "error": print("\nERROR:", m["msg"]); done.set(); return
        rt = asyncio.create_task(recv())
        await ws.send(json.dumps({"type": "start"}))
        await asyncio.sleep(0.3)
        for i in range(0, len(audio), CHUNK):
            await ws.send(audio[i:i+CHUNK].tobytes())   # binaire PCM16-frame
            await asyncio.sleep(CHUNK/sr)               # real-time tempo
        await ws.send(json.dumps({"type": "stop"}))
        try: await asyncio.wait_for(done.wait(), timeout=15)
        except asyncio.TimeoutError: print("\n[timeout]")
        rt.cancel()
    print(f"\n=> {'OK' if got else 'GEEN TEKST'} ({len(''.join(got))} tekens)")

asyncio.run(main())
