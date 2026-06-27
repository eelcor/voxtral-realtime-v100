#!/usr/bin/env python3
"""Score one Whisper model's WER, either on the prepped sweep clips (--dir) or on
N freshly streamed FLEURS-nl clips (--n). Writes <dir>/whisper_<model>.json when
run against a clips dir, so plot_delay.py can pick it up as a baseline line.
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", default=None, help="prepped clips dir (uses meta.json)")
    ap.add_argument("--n", type=int, default=80, help="stream N clips if --dir not given")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from faster_whisper import WhisperModel
    print(f"Whisper laden: {args.model}", flush=True)
    wm = WhisperModel(args.model, device="cuda", compute_type="float16")

    def tx(path):
        segs, _ = wm.transcribe(path, language="nl", beam_size=5)
        return " ".join(s.text for s in segs).strip()

    refs, hyps = [], []
    if args.dir:
        meta = json.load(open(os.path.join(args.dir, "meta.json")))
        for i, c in enumerate(meta["clips"]):
            refs.append(normalize(c["ref"])); hyps.append(normalize(tx(c["file"])))
            print(f"[{i+1}/{len(meta['clips'])}]", flush=True)
        n = len(meta["clips"])
    else:
        from datasets import load_dataset, Audio
        ds = load_dataset("google/fleurs", "nl_nl", split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        tmp = "/tmp/_we_clip.wav"
        it = iter(ds)
        for i in range(args.n):
            ex = next(it)
            arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            sf.write(tmp, arr, sr, subtype="PCM_16")
            refs.append(normalize(ex["transcription"])); hyps.append(normalize(tx(tmp)))
            print(f"[{i+1}/{args.n}]", flush=True)
        n = args.n

    w = wer(refs, hyps)
    print(f"\nWhisper {args.model} WER = {100*w:.1f}%  ({n} clips)")
    if args.dir:
        out = os.path.join(args.dir, f"whisper_{args.model}.json")
        json.dump({"whisper_model": args.model, "whisper_wer": w, "clips": n},
                  open(out, "w"), ensure_ascii=False, indent=2)
        print("geschreven:", out)


if __name__ == "__main__":
    main()
