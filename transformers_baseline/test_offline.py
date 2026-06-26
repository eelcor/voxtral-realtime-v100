#!/usr/bin/env python3
"""Validatie: stream een bekend spraakfragment door de Voxtral-realtime pipeline.

Bewijst dat model + streaming-generate op de V100 correcte transcriptie geeft,
los van de microfoon. Gebruikt een Engels LibriSpeech-sample als referentie.
"""
import sys
import threading
import time

import io

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import (
    TextIteratorStreamer,
    VoxtralRealtimeForConditionalGeneration,
    VoxtralRealtimeProcessor,
)

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
DEV = "cuda:0"

print(f"[laden] {MODEL_ID} ...", file=sys.stderr, flush=True)
processor = VoxtralRealtimeProcessor.from_pretrained(MODEL_ID)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map=DEV
).eval()

# decode=False -> geen torchcodec nodig; we lezen de flac zelf met soundfile
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))
sample = ds[0]
a = sample["audio"]
raw = a.get("bytes")
audio, sr = sf.read(io.BytesIO(raw)) if raw else sf.read(a["path"])
audio = np.asarray(audio, dtype=np.float32)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
assert sr == 16000, f"verwacht 16kHz, kreeg {sr}"
print(f"[audio] {len(audio)/16000:.1f}s @{sr}Hz", file=sys.stderr)
print(f"[referentie] {sample['text']}", file=sys.stderr, flush=True)

# rechts opvullen zoals de docs voorschrijven
audio = np.pad(audio, (0, processor.num_right_pad_tokens() * processor.raw_audio_length_per_tok))

hop = processor.feature_extractor.hop_length
win = processor.feature_extractor.win_length

first_inputs = processor(audio[:processor.num_samples_first_audio_chunk],
                         is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")

def gen():
    yield first_inputs.input_features.to(DEV, torch.float16)
    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    start_idx = mel_frame_idx * hop - win // 2
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
        inp = processor(audio[start_idx:end_idx], is_streaming=True,
                        is_first_audio_chunk=False, return_tensors="pt")
        yield inp.input_features.to(DEV, torch.float16)
        mel_frame_idx += processor.audio_length_per_tok
        start_idx = mel_frame_idx * hop - win // 2

streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
kwargs = dict(input_ids=first_inputs.input_ids.to(DEV), input_features=gen(),
              num_delay_tokens=first_inputs.num_delay_tokens, streamer=streamer,
              do_sample=False, max_new_tokens=2000)
t0 = time.time()
th = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
th.start()
print("[transcriptie] ", end="", file=sys.stderr, flush=True)
out = []
for txt in streamer:
    out.append(txt)
    print(txt, end="", flush=True)
th.join()
print(f"\n[klaar] {time.time()-t0:.1f}s wall, {len(audio)/16000:.1f}s audio", file=sys.stderr)
