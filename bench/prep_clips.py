#!/usr/bin/env python3
"""Dump N FLEURS-nl clips to disk once, with the (fixed) Whisper baseline.

So every delay setting in the sweep transcribes the EXACT same audio, and Whisper
(which is delay-independent) only runs once. Writes <dir>/clip_*.wav + meta.json.
"""
import argparse
import io
import json
import os
import re

import soundfile as sf
from jiwer import wer


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dir", default="/tmp/sweep_clips")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", "nl_nl", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    from faster_whisper import WhisperModel
    print(f"Whisper laden: {args.whisper_model}", flush=True)
    wm = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")

    meta, refs, whyps = [], [], []
    it = iter(ds)
    for i in range(args.n):
        ex = next(it)
        arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        path = os.path.join(args.dir, f"clip_{i:03d}.wav")
        sf.write(path, arr, sr, subtype="PCM_16")
        ref = ex["transcription"]
        segs, _ = wm.transcribe(path, language="nl", beam_size=5)
        whyp = " ".join(s.text for s in segs).strip()
        meta.append({"file": path, "dur": round(len(arr) / sr, 2), "ref": ref, "whisper": whyp})
        refs.append(normalize(ref)); whyps.append(normalize(whyp))
        print(f"[{i+1}/{args.n}] {meta[-1]['dur']}s", flush=True)

    whisper_wer = wer(refs, whyps)
    with open(os.path.join(args.dir, "meta.json"), "w") as f:
        json.dump({"whisper_model": args.whisper_model,
                   "whisper_wer": whisper_wer, "clips": meta}, f, ensure_ascii=False, indent=2)
    print(f"\nWhisper {args.whisper_model} WER = {100*whisper_wer:.1f}%  ({args.n} clips)")
    print(f"geschreven: {args.dir}/meta.json")


if __name__ == "__main__":
    main()
