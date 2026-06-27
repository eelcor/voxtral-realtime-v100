#!/usr/bin/env python3
"""Real-life test: knip een stuk uit een (m4a/mp3/wav) opname en transcribeer het
met Voxtral-realtime én Whisper, side-by-side. Geen referentie -> geen WER, puur
kwalitatief vergelijken.

  python realtest.py ../realtest/opname.m4a --start 0 --dur 60
"""
import argparse
import asyncio
import os
import time

import soundfile as sf
from faster_whisper.audio import decode_audio

import voxtral_file

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--start", type=float, default=0.0, help="startseconde van het fragment")
    ap.add_argument("--dur", type=float, default=60.0, help="duur in seconden (0 = volledig)")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    audio = decode_audio(args.audio, sampling_rate=SR)  # 16k mono float32 (PyAV)
    total = len(audio) / SR
    a = int(args.start * SR)
    b = len(audio) if args.dur <= 0 else min(len(audio), a + int(args.dur * SR))
    clip = audio[a:b]
    clip_s = len(clip) / SR
    wav = "/tmp/realtest_clip.wav"
    sf.write(wav, clip, SR, subtype="PCM_16")
    print(f"opname: {total:.0f}s totaal | fragment: {args.start:.0f}–{args.start+clip_s:.0f}s ({clip_s:.0f}s)\n")

    from faster_whisper import WhisperModel
    wm = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
    t0 = time.time()
    segs, _ = wm.transcribe(wav, language="nl", beam_size=5)
    wtext = " ".join(s.text for s in segs).strip()
    wt = time.time() - t0

    t0 = time.time()
    vtext = asyncio.run(voxtral_file.transcribe(wav))
    vt = time.time() - t0

    print(f"=== Whisper {args.whisper_model}  ({wt:.1f}s) ===\n{wtext}\n")
    print(f"=== Voxtral-realtime  ({vt:.1f}s) ===\n{vtext}\n")


if __name__ == "__main__":
    main()
