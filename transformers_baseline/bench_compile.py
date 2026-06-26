#!/usr/bin/env python3
"""Compileer alleen de audio-encoder (vaste shape per stap) en meet de winst.
Print elk resultaat direct (flushed)."""
import io, sys, time, threading
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import (TextIteratorStreamer,
                          VoxtralRealtimeForConditionalGeneration,
                          VoxtralRealtimeProcessor)

DEV = "cuda:0"
MID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
SR = 16000

def log(*a): print(*a, flush=True)

log("[laden] sdpa ...")
proc = VoxtralRealtimeProcessor.from_pretrained(MID)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MID, dtype=torch.float16, device_map=DEV, attn_implementation="sdpa").eval()
hop = proc.feature_extractor.hop_length; win = proc.feature_extractor.win_length
step_ms = proc.audio_length_per_tok * hop / SR * 1000

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))
a = ds[0]["audio"]
base, sr = sf.read(io.BytesIO(a["bytes"])) if a.get("bytes") else sf.read(a["path"])
base = np.asarray(base, dtype=np.float32)

def build(audio):
    first = proc(audio[:proc.num_samples_first_audio_chunk], is_streaming=True,
                 is_first_audio_chunk=True, return_tensors="pt")
    feats = [first.input_features.to(DEV, torch.float16)]
    mel = proc.num_mel_frames_first_audio_chunk; start = mel*hop - win//2
    while (end := start + proc.num_samples_per_audio_chunk) < audio.shape[0]:
        inp = proc(audio[start:end], is_streaming=True, is_first_audio_chunk=False, return_tensors="pt")
        feats.append(inp.input_features.to(DEV, torch.float16))
        mel += proc.audio_length_per_tok; start = mel*hop - win//2
    return first, feats

def run(first, feats):
    times=[]
    def gen():
        for i,f in enumerate(feats):
            t=time.time(); yield f
            if i>0: times.append(time.time()-t)
    streamer = TextIteratorStreamer(proc.tokenizer, skip_special_tokens=True)
    kw = dict(input_ids=first.input_ids.to(DEV), input_features=gen(),
              num_delay_tokens=first.num_delay_tokens, streamer=streamer,
              do_sample=False, max_new_tokens=100000)
    drain = threading.Thread(target=lambda:[None for _ in streamer]); drain.start()
    t0=time.time(); model.generate(**kw); wall=time.time()-t0; drain.join()
    return wall, np.array(times)

def report(tag, wall, times, dur):
    steady = times[3:] if len(times)>6 else times
    med = np.median(steady)*1000 if len(steady) else float("nan")
    log(f"  {tag:20s} RTF {wall/dur:.2f} | per stap mediaan {med:.0f}ms "
        f"(budget {step_ms:.0f}ms) -> stap-RTF {med/step_ms:.2f} "
        f"{'BIJ' if med<step_ms else 'ACHTER'}")

long_audio = np.tile(base, 5); dur = len(long_audio)/sr   # ~29s
log(f"[audio] meting {dur:.0f}s")

log("=== BASELINE ===")
f1,t1 = build(long_audio); w,t = run(f1,t1); report("baseline", w, t, dur)

log("=== compile audio_tower (default mode) ===")
log("  [compileren+warmup...] ")
model.model.audio_tower = torch.compile(model.model.audio_tower, mode="default", fullgraph=False)
fw,tw = build(np.tile(base,2))
try:
    run(fw,tw); run(fw,tw)
    log("  [warmup klaar]")
except Exception as e:
    log(f"  [warmup faalde] {type(e).__name__}: {e}")
f2,t2 = build(long_audio)
try:
    w2,t2 = run(f2,t2); report("tower-compiled", w2, t2, dur)
except Exception as e:
    log(f"  [run faalde] {type(e).__name__}: {e}")
log("[klaar]")
