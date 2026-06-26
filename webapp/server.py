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
import audioop
import base64
import json
import os
import urllib.request

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

VOXTRAL_WS = os.environ.get("VOXTRAL_WS", "ws://127.0.0.1:8045/v1/realtime")
VOXTRAL_HEALTH = os.environ.get("VOXTRAL_HEALTH", "http://127.0.0.1:8045/health")
MODEL = os.environ.get("VOXTRAL_MODEL", "voxtral-realtime")

# Klein LLM (llama-server, OpenAI-compatibel) voor samenvatting / formulier-extractie
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8052/v1/chat/completions")
LLM_HEALTH = os.environ.get("LLM_HEALTH", "http://127.0.0.1:8052/health")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma")
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

# ~600 ms stilte om de laatste woorden uit de delay-buffer te flushen bij stop
SILENCE_B64 = base64.b64encode(bytes(2 * 9600)).decode()

# Sessie-rollover: vóór de contextlimiet (max_model_len ~3072 tokens ≈ 245s audio)
# de Voxtral-sessie naadloos herstarten, zodat lange opnames niet crashen.
# Zacht: vanaf hier rollen op de eerstvolgende STILTE (natuurlijke zingrens).
# Hard: uiterlijk hier rollen, ongeacht spraak (veiligheid vóór de crash).
ROLLOVER_S = float(os.environ.get("VOXTRAL_ROLLOVER_S", "150"))
ROLLOVER_HARD_S = float(os.environ.get("VOXTRAL_ROLLOVER_HARD_S", "210"))
# Een echte zin-pauze = aanhoudende stilte van minstens zoveel seconden,
# gemeten t.o.v. een adaptieve ruisvloer (past zich aan de omgeving aan).
MIN_PAUSE_S = float(os.environ.get("VOXTRAL_MIN_PAUSE_S", "0.6"))
# Absolute minimum-marge boven de ruisvloer (0..32768) om "stil" te bepalen.
SILENCE_RMS = float(os.environ.get("VOXTRAL_SILENCE_RMS", "250"))

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


@app.get("/llm/health")
async def llm_health():
    try:
        urllib.request.urlopen(LLM_HEALTH, timeout=3)
        return {"llm": "ready"}
    except Exception as e:
        return JSONResponse({"llm": "down", "error": str(e)[:120]}, status_code=503)


def _llm_call(body):
    req = urllib.request.Request(LLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _parse_json(txt):
    txt = (txt or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt[:4].lower() == "json":
            txt = txt[4:]
    i, j = txt.find("{"), txt.rfind("}")
    if i != -1 and j != -1:
        txt = txt[i:j + 1]
    return json.loads(txt)


@app.post("/process")
async def process(req: Request):
    data = await req.json()
    mode = data.get("mode", "summary")
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return {"mode": mode, "result": ({} if mode == "form" else "")}

    if mode == "form":
        fields = data.get("fields") or []
        keys = [f["name"] for f in fields]
        sys = ("Je extraheert gestructureerde gegevens uit een (mogelijk onvolledige) transcriptie van spraak. "
               "Antwoord UITSLUITEND met één JSON-object met exact deze sleutels: " + ", ".join(keys) + ". "
               "Vul elke waarde op basis van wat expliciet gezegd is; gebruik een lege string \"\" als iets niet "
               "wordt genoemd. Elke waarde is een korte tekststring. Verzin niets. "
               "Geen uitleg en geen markdown — alleen het JSON-object.")
        flist = "\n".join(f"- {f['name']}: {f.get('description','')}" for f in fields)
        user = f"Transcriptie:\n{transcript}\n\nVelden:\n{flist}"
        body = {"model": LLM_MODEL,
                "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
                "temperature": 0.1, "max_tokens": 700,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            out = await asyncio.to_thread(_llm_call, body)
            result = _parse_json(out["choices"][0]["message"]["content"])
        except Exception as e:
            return JSONResponse({"mode": "form", "error": str(e)[:200]}, status_code=502)
        return {"mode": "form", "result": result}

    sys = ("Je bent een beknopte notulist. Vat de volgende (mogelijk onvolledige) transcriptie van spraak "
           "samen in korte, heldere bullets in het Nederlands. Alleen de kernpunten; geen inleiding of afsluiting.")
    body = {"model": LLM_MODEL,
            "messages": [{"role": "system", "content": sys}, {"role": "user", "content": transcript}],
            "temperature": 0.3, "max_tokens": 450,
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        out = await asyncio.to_thread(_llm_call, body)
        txt = out["choices"][0]["message"]["content"]
    except Exception as e:
        return JSONResponse({"mode": "summary", "error": str(e)[:200]}, status_code=502)
    return {"mode": "summary", "result": txt}


@app.post("/stream")
async def stream(req: Request):
    """Streamt een live samenvatting token-voor-token (lagere ervaren latency)."""
    data = await req.json()
    transcript = (data.get("transcript") or "").strip()
    sys = ("Je bent een beknopte notulist. Vat de (mogelijk onvolledige) transcriptie samen in maximaal "
           "5 korte bullets in het Nederlands. Alleen de kernpunten; geen inleiding of afsluiting.")
    body = {"model": LLM_MODEL,
            "messages": [{"role": "system", "content": sys}, {"role": "user", "content": transcript}],
            "temperature": 0.3, "max_tokens": 300, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}

    async def gen():
        if not transcript:
            return
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                async with c.stream("POST", LLM_URL, json=body) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        d = line[5:].strip()
                        if d == "[DONE]":
                            break
                        try:
                            delta = json.loads(d)["choices"][0]["delta"].get("content", "")
                        except Exception:
                            continue
                        if delta:
                            yield delta
        except Exception:
            return

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.websocket("/ws")
async def ws(client: WebSocket):
    await client.accept()
    up = None
    relay_task = None
    started = False
    audio_s = 0.0       # seconden audio in de huidige Voxtral-sessie
    noise_floor = None  # adaptieve ruisvloer (RMS), volgt de stilte-envelope
    quiet_run = 0.0      # aaneengesloten seconden stilte
    seg_done = None      # Event: gezet zodra de huidige sessie transcription.done stuurt

    async def relay(conn, done_event):
        try:
            async for msg in conn:
                ev = json.loads(msg)
                t = ev.get("type")
                if t == "transcription.delta":
                    await client.send_json({"type": "delta", "text": ev.get("delta", "")})
                elif t == "transcription.done":
                    await client.send_json({"type": "segment_done", "text": ev.get("text", "")})
                    done_event.set()
                elif t == "error":
                    await client.send_json({"type": "error", "msg": str(ev.get("error", ev))})
        except Exception:
            pass

    async def open_session(prime_silence=False):
        nonlocal up, relay_task, started, audio_s, seg_done
        up = await websockets.connect(VOXTRAL_WS, max_size=None, ping_interval=None)
        await up.send(json.dumps({"type": "session.update", "model": MODEL}))
        started = False
        audio_s = 0.0
        seg_done = asyncio.Event()
        relay_task = asyncio.create_task(relay(up, seg_done))
        if prime_silence:
            # Prime de nieuwe sessie op stilte: het model gebruikt z'n eerste
            # ~562ms als priming-venster, dus dat moet stilte zijn i.p.v. het
            # eerste echte woord (anders verdwijnt dat woord op de naad).
            await up.send(json.dumps({"type": "input_audio_buffer.append", "audio": SILENCE_B64}))
            await up.send(json.dumps({"type": "input_audio_buffer.commit"}))
            started = True

    async def finalize_session(wait_done=2.5):
        # Sluit de sessie af: flush met stilte, commit final, en WACHT op de
        # transcription.done zodat de laatste woorden gegarandeerd zijn doorgestuurd.
        nonlocal up, relay_task, started, seg_done
        if up is not None:
            try:
                # Ruime flush-stilte (zit in de queue als 'final' komt) zodat de
                # generatie de ~480ms output-delaybuffer leegmaakt = laatste woord.
                for _ in range(3):
                    await up.send(json.dumps({"type": "input_audio_buffer.append", "audio": SILENCE_B64}))
                await up.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
                if wait_done > 0 and seg_done is not None:
                    try:
                        await asyncio.wait_for(seg_done.wait(), timeout=wait_done)
                    except asyncio.TimeoutError:
                        pass
            except Exception:
                pass
            try:
                await up.close()
            except Exception:
                pass
        up = None
        started = False
        seg_done = None
        if relay_task:
            relay_task.cancel()
            relay_task = None

    try:
        while True:
            m = await client.receive()
            if m.get("type") == "websocket.disconnect":
                break

            txt = m.get("text")
            byt = m.get("bytes")

            if txt is not None:
                kind = json.loads(txt).get("type")
                if kind == "start":
                    await finalize_session(wait_done=0)
                    try:
                        await open_session()
                    except Exception as e:
                        await client.send_json({"type": "error", "msg": f"Kan Voxtral niet bereiken: {e}"})
                        continue
                    await client.send_json({"type": "ready"})
                elif kind == "stop":
                    await finalize_session()

            elif byt is not None and up is not None:
                # Rollover vóór de contextlimiet: bij voorkeur op een ECHTE
                # zin-pauze (aanhoudende stilte t.o.v. een adaptieve ruisvloer)
                # zodra de zachte drempel is bereikt; uiterlijk bij de harde grens.
                try:
                    rms = audioop.rms(byt, 2)
                except Exception:
                    rms = 9999
                # adaptieve ruisvloer: snel zakken naar lage waarden, traag stijgen
                if noise_floor is None:
                    noise_floor = rms
                elif rms < noise_floor:
                    noise_floor = 0.9 * noise_floor + 0.1 * rms
                else:
                    noise_floor = 0.995 * noise_floor + 0.005 * rms
                frame_s = len(byt) / 2 / 16000.0
                quiet = rms < noise_floor * 2.0 + SILENCE_RMS
                quiet_run = quiet_run + frame_s if quiet else 0.0
                if (audio_s >= ROLLOVER_S and quiet_run >= MIN_PAUSE_S) or audio_s >= ROLLOVER_HARD_S:
                    quiet_run = 0.0
                    await finalize_session()
                    try:
                        await open_session(prime_silence=True)
                    except Exception as e:
                        await client.send_json({"type": "error", "msg": f"rollover faalde: {e}"})
                        continue
                try:
                    audio_s += len(byt) / 2 / 16000.0
                    b64 = base64.b64encode(byt).decode()
                    await up.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                    if not started:
                        await up.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        started = True
                except Exception as e:
                    await client.send_json({"type": "error", "msg": f"stream onderbroken: {e}"})
                    await finalize_session(wait_done=0)
    except WebSocketDisconnect:
        pass
    finally:
        await finalize_session(wait_done=0)
