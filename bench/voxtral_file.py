#!/usr/bin/env python3
"""Transcribeer een audiobestand via de directe Voxtral realtime WebSocket (vLLM).

De realtime-API is streaming-only, dus we spelen het bestand als een stream af:
session.update -> audio in chunks appenden -> wat stilte om de delay-buffer te
flushen -> commit(final) -> wachten op transcription.done.

Gebruik:  python voxtral_file.py audio.wav
Of importeer:  from voxtral_file import transcribe
"""
import asyncio
import base64
import json
import os
import sys

import numpy as np
import soundfile as sf
import websockets

VOXTRAL_WS = os.environ.get("VOXTRAL_WS", "ws://127.0.0.1:8045/v1/realtime")
MODEL = os.environ.get("VOXTRAL_MODEL", "voxtral-realtime")
SR = 16000


def load_pcm16(path):
    """Laad een audiobestand als 16 kHz mono PCM16 (int16, little-endian)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2")


async def transcribe(path, model=MODEL, ws_url=VOXTRAL_WS, tail_silence_s=2.0,
                     drain_timeout=2.5, max_wait=180.0):
    """Burst-feed het bestand en lees door tot de output opdroogt.

    Cruciaal: een gewone `commit` (geen `final`) zodat de toegevoegde stilte óók
    wordt gedecodeerd en de ~480ms output-delaybuffer leegloopt — anders kapt het
    model het laatste woord af. We breken niet af op de eerste `done`, maar lezen
    tot er `drain_timeout` seconden niets meer binnenkomt.
    """
    pcm = load_pcm16(path)
    silence = np.zeros(int(SR * tail_silence_s), dtype="<i2")
    chunk = SR // 10  # 100 ms

    def append(buf):
        return json.dumps({"type": "input_audio_buffer.append",
                           "audio": base64.b64encode(buf.tobytes()).decode()})

    deltas, done_texts = [], []
    async with websockets.connect(ws_url, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        started = False
        for i in range(0, len(pcm), chunk):
            await ws.send(append(pcm[i:i + chunk]))
            if not started:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                started = True
        await ws.send(append(silence))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))  # flush, niet final

        loop = asyncio.get_event_loop()
        deadline = loop.time() + max_wait
        while loop.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=drain_timeout)
            except asyncio.TimeoutError:
                break  # output opgedroogd
            m = json.loads(msg)
            t = m.get("type")
            if t == "transcription.delta":
                deltas.append(m.get("delta", ""))
            elif t == "transcription.done":
                done_texts.append(m.get("text", ""))
            elif t == "error":
                raise RuntimeError(m.get("error", "voxtral error"))
        try:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        except Exception:
            pass

    # deltas-concatenatie is robuust tegen meerdere segmenten; done_texts als fallback
    text = "".join(deltas).strip()
    if not text and done_texts:
        text = " ".join(done_texts).strip()
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("gebruik: python voxtral_file.py audio.wav", file=sys.stderr)
        sys.exit(1)
    print(asyncio.run(transcribe(sys.argv[1])))
