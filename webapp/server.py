#!/usr/bin/env python3
"""Backend voor de Voxtral realtime-transcriptie-webapp.

Serveert de statische app (HTTPS) en proxiet de browser-audio naar de vLLM
/v1/realtime websocket. Eén poort, geen vLLM-poort naar buiten exposen.

Browser <-> backend protocol (op /ws):
  browser -> backend:
    text  {"type":"start"}              # begin opname-sessie
    bytes <Int16LE PCM @16kHz frame>    # audio
    text  {"type":"stop"}               # einde opname-sessie
  backend -> browser:
    {"type":"ready"}                    # vLLM-sessie open
    {"type":"delta","text":"..."}       # incrementele transcriptie
    {"type":"segment_done","text":"..."}# segment afgerond
    {"type":"error","msg":"..."}
"""
import asyncio
import base64
import json
import os

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

VOXTRAL_WS = os.environ.get("VOXTRAL_WS", "ws://127.0.0.1:8045/v1/realtime")
VOXTRAL_HEALTH = os.environ.get("VOXTRAL_HEALTH", "http://127.0.0.1:8045/health")
MODEL = os.environ.get("VOXTRAL_MODEL", "voxtral-realtime")
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

# ~600 ms stilte om de laatste woorden uit de delay-buffer te flushen bij stop
SILENCE_B64 = base64.b64encode(bytes(2 * 9600)).decode()

app = FastAPI(title="Voxtral Realtime")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/healthz")
async def healthz():
    import urllib.request
    try:
        urllib.request.urlopen(VOXTRAL_HEALTH, timeout=3)
        return {"model": "ready"}
    except Exception as e:
        return JSONResponse({"model": "down", "error": str(e)[:120]}, status_code=503)


@app.websocket("/ws")
async def ws(client: WebSocket):
    await client.accept()
    up = None
    relay_task = None
    started = False

    async def relay(conn):
        try:
            async for msg in conn:
                ev = json.loads(msg)
                t = ev.get("type")
                if t == "transcription.delta":
                    await client.send_json({"type": "delta", "text": ev.get("delta", "")})
                elif t == "transcription.done":
                    await client.send_json({"type": "segment_done", "text": ev.get("text", "")})
                elif t == "error":
                    await client.send_json({"type": "error", "msg": str(ev.get("error", ev))})
        except Exception:
            pass

    async def close_up():
        nonlocal up, relay_task, started
        if up is not None:
            try:
                await up.close()
            except Exception:
                pass
        up = None
        started = False
        if relay_task:
            relay_task.cancel()

    try:
        while True:
            m = await client.receive()
            if m.get("type") == "websocket.disconnect":
                break

            txt = m.get("text")
            byt = m.get("bytes")

            if txt is not None:
                data = json.loads(txt)
                kind = data.get("type")
                if kind == "start":
                    await close_up()
                    try:
                        up = await websockets.connect(VOXTRAL_WS, max_size=None, ping_interval=None)
                    except Exception as e:
                        await client.send_json({"type": "error", "msg": f"Kan Voxtral niet bereiken: {e}"})
                        continue
                    await up.send(json.dumps({"type": "session.update", "model": MODEL}))
                    started = False
                    relay_task = asyncio.create_task(relay(up))
                    await client.send_json({"type": "ready"})
                elif kind == "stop":
                    if up is not None:
                        try:
                            await up.send(json.dumps({"type": "input_audio_buffer.append", "audio": SILENCE_B64}))
                            await up.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
                            await asyncio.sleep(1.0)  # laat de laatste deltas binnenkomen
                        except Exception:
                            pass
                    await close_up()

            elif byt is not None and up is not None:
                try:
                    b64 = base64.b64encode(byt).decode()
                    await up.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                    if not started:
                        await up.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        started = True
                except Exception as e:
                    await client.send_json({"type": "error", "msg": f"stream onderbroken: {e}"})
                    await close_up()
    except WebSocketDisconnect:
        pass
    finally:
        await close_up()
