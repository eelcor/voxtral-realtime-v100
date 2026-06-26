#!/usr/bin/env python3
"""Meet of Voxtral-realtime de audio real-time kan bijhouden op deze V100.

Voert een lang spraakfragment door de streaming-pipeline en meet:
  - RTF (real-time factor) = verwerkingstijd / audioduur.  <1 = houdt bij, >1 = loopt achter.
  - per-chunk verwerkingstijd t.o.v. de chunkduur (105ms).
Geen mic nodig.
"""
import io
import sys
import time

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ATTN = sys.argv[3] if len(sys.argv) > 3 else "sdpa"

print(f"[laden] op {DEV} attn={ATTN} ...", file=sys.stderr, flush=True)
processor = VoxtralRealtimeProcessor.from_pretrained("mistralai/Voxtral-Mini-4B-Realtime-2602")
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    "mistralai/Voxtral-Mini-4B-Realtime-2602", dtype=torch.float16, device_map=DEV,
    attn_implementation=ATTN).eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))
a = ds[0]["audio"]
audio, sr = sf.read(io.BytesIO(a["bytes"])) if a.get("bytes") else sf.read(a["path"])
audio = np.asarray(audio, dtype=np.float32)
audio = np.tile(audio, REPEAT)  # langer maken
dur = len(audio) / sr
print(f"[audio] {dur:.1f}s (sample {REPEAT}x herhaald)", file=sys.stderr)

hop = processor.feature_extractor.hop_length
win = processor.feature_extractor.win_length
chunk_ms = processor.num_samples_per_audio_chunk / sr * 1000
# Echte budget = hoeveel audio per stap opschuift (chunks overlappen!): 8*160/16000 = 80ms
step_ms = processor.audio_length_per_tok * hop / sr * 1000

first = processor(audio[:processor.num_samples_first_audio_chunk], is_streaming=True,
                  is_first_audio_chunk=True, return_tensors="pt")

# Bouw alle chunk-features vooraf (CPU), zodat we puur de GPU-stap timen.
feats = [first.input_features.to(DEV, torch.float16)]
mel = processor.num_mel_frames_first_audio_chunk
start = mel * hop - win // 2
while (end := start + processor.num_samples_per_audio_chunk) < audio.shape[0]:
    inp = processor(audio[start:end], is_streaming=True, is_first_audio_chunk=False,
                    return_tensors="pt")
    feats.append(inp.input_features.to(DEV, torch.float16))
    mel += processor.audio_length_per_tok
    start = mel * hop - win // 2

print(f"[chunks] {len(feats)} stuks van ~{chunk_ms:.0f}ms", file=sys.stderr, flush=True)

# Streaming generate met een generator die per chunk de verwerkingstijd logt.
from transformers import TextIteratorStreamer
import threading

times = []
def gen():
    for i, f in enumerate(feats):
        t = time.time()
        yield f
        if i > 0:
            times.append(time.time() - t)  # tijd die generate per chunk nam

streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
ntok = [0]
def count():
    for txt in streamer:
        ntok[0] += len(processor.tokenizer.encode(txt)) if txt else 0

kw = dict(input_ids=first.input_ids.to(DEV), input_features=gen(),
          num_delay_tokens=first.num_delay_tokens, streamer=streamer,
          do_sample=False, max_new_tokens=100000)
t0 = time.time()
gt = threading.Thread(target=model.generate, kwargs=kw); gt.start()
ct = threading.Thread(target=count); ct.start()
gt.join(); ct.join()
wall = time.time() - t0

# Warmup (eerste paar chunks) negeren voor steady-state.
steady = np.array(times[3:]) if len(times) > 6 else np.array(times)
print("\n=== RESULTAAT ===")
print(f"audioduur        : {dur:.1f} s")
print(f"verwerkingstijd  : {wall:.1f} s")
print(f"RTF (totaal)     : {wall/dur:.2f}   ({'HOUDT BIJ' if wall<dur else 'LOOPT ACHTER'})")
if len(steady):
    print(f"per stap (ms)    : mediaan {np.median(steady)*1000:.0f}  p90 {np.percentile(steady,90)*1000:.0f}  "
          f"max {np.max(steady)*1000:.0f}   (budget {step_ms:.0f}ms = audio/stap)")
    print(f"stap-RTF         : {np.median(steady)*1000/step_ms:.2f} (mediaan; <1 = haalbaar)")
print(f"tokens (ca.)     : {ntok[0]}  ->  {ntok[0]/wall:.1f} tok/s")
