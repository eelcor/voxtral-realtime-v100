#!/usr/bin/env python3
"""Transcribe the prepped clips with the CURRENTLY RUNNING Voxtral server and
score WER. The server's delay is whatever it was launched with; we read the
actual value from /tmp/voxtral_delay.json so the point is labelled correctly.
"""
import argparse
import asyncio
import json
import os
import re
import time

from jiwer import wer

import voxtral_file


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/sweep_clips")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    delay = {}
    if os.path.exists("/tmp/voxtral_delay.json"):
        delay = json.load(open("/tmp/voxtral_delay.json"))
    n_tok = delay.get("num_delay_tokens", 6)
    delay_ms = delay.get("transcription_delay_ms", 480.0)

    refs, vhyps = [], []
    audio_s = vox_t = 0.0
    for i, c in enumerate(meta["clips"]):
        t0 = time.time()
        v = asyncio.run(voxtral_file.transcribe(c["file"]))
        vox_t += time.time() - t0
        audio_s += c["dur"]
        refs.append(normalize(c["ref"])); vhyps.append(normalize(v))
        print(f"[{i+1}/{len(meta['clips'])}] delay={n_tok}tok  WER={100*wer(normalize(c['ref']), normalize(v)):5.1f}%", flush=True)

    vox_wer = wer(refs, vhyps)
    res = {"num_delay_tokens": n_tok, "delay_ms": delay_ms,
           "wer_voxtral": vox_wer, "clips": len(meta["clips"]),
           "audio_seconds": round(audio_s, 1)}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\ndelay={n_tok}tok ({delay_ms:.0f}ms)  Voxtral WER={100*vox_wer:.1f}%  -> {args.out}")


if __name__ == "__main__":
    main()
