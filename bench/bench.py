#!/usr/bin/env python3
"""WER-benchmark: Whisper vs. Voxtral-realtime op Nederlandse spraak.

Pakt N clips uit FLEURS-nl (voorgelezen spraak met referentie-transcript),
transcribeert ze met faster-whisper en met de draaiende Voxtral-realtime-server,
en rekent de Word Error Rate uit. Schrijft results.json + results.md.

Voorbeelden:
  python bench.py --n 30 --whisper-model large-v3
  python bench.py --n 50 --whisper-model medium --device cuda --gpu 0
  python bench.py --audio mijn_opname.wav        # eigen audio, side-by-side (geen WER)
"""
import argparse
import asyncio
import json
import os
import re
import time

import numpy as np
import soundfile as sf
from jiwer import wer

import voxtral_file


def normalize(s):
    """Lichte normalisatie voor een eerlijke WER: kleine letters, leestekens weg,
    spaties samenvouwen. (\\w dekt ook accenten/cijfers in unicode.)"""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def fmt_pct(x):
    return f"{100 * x:5.1f}%"


def load_fleurs(n):
    # decode=False -> geen torchcodec nodig; we lezen de bytes zelf met soundfile
    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", "nl_nl", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for ex in ds:
        out.append(ex)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="aantal clips")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES voor Whisper")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--audio", default=None, help="eigen wav: side-by-side, geen WER")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ctype = args.compute_type if args.device == "cuda" else "int8"

    from faster_whisper import WhisperModel
    print(f"Whisper laden: {args.whisper_model} ({args.device}/{ctype}) ...", flush=True)
    wmodel = WhisperModel(args.whisper_model, device=args.device, compute_type=ctype)

    def whisper_tx(path):
        segs, _ = wmodel.transcribe(path, language="nl", beam_size=args.beam)
        return " ".join(s.text for s in segs).strip()

    def voxtral_tx(path):
        return asyncio.run(voxtral_file.transcribe(path))

    # --- eigen audio: alleen side-by-side ---
    if args.audio:
        t0 = time.time(); w = whisper_tx(args.audio); tw = time.time() - t0
        t0 = time.time(); v = voxtral_tx(args.audio); tv = time.time() - t0
        print("\n=== eigen audio ===")
        print(f"Whisper  ({tw:4.1f}s): {w}")
        print(f"Voxtral  ({tv:4.1f}s): {v}")
        return

    # --- FLEURS WER ---
    print(f"FLEURS-nl laden ({args.n} clips) ...", flush=True)
    samples = load_fleurs(args.n)

    rows, refs, w_hyps, v_hyps = [], [], [], []
    w_audio_s = w_time = v_time = 0.0
    tmp = "/tmp/_bench_clip.wav"
    import io
    for i, ex in enumerate(samples):
        arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        sf.write(tmp, arr, sr, subtype="PCM_16")
        dur = len(arr) / sr
        ref = ex["transcription"]

        t0 = time.time(); w = whisper_tx(tmp); w_time += time.time() - t0
        t0 = time.time(); v = voxtral_tx(tmp); v_time += time.time() - t0
        w_audio_s += dur

        nref, nw, nv = normalize(ref), normalize(w), normalize(v)
        refs.append(nref); w_hyps.append(nw); v_hyps.append(nv)
        rw, rv = wer(nref, nw), wer(nref, nv)
        rows.append({"i": i, "dur": round(dur, 1), "ref": ref,
                     "whisper": w, "voxtral": v,
                     "wer_whisper": rw, "wer_voxtral": rv})
        print(f"[{i+1}/{len(samples)}] {dur:4.1f}s  W={fmt_pct(rw)}  V={fmt_pct(rv)}", flush=True)

    corp_w = wer(refs, w_hyps)
    corp_v = wer(refs, v_hyps)
    summary = {
        "clips": len(samples),
        "audio_seconds": round(w_audio_s, 1),
        "whisper_model": args.whisper_model,
        "wer_whisper": corp_w,
        "wer_voxtral": corp_v,
        "rtf_whisper": round(w_time / max(w_audio_s, 1e-9), 3),
        "rtf_voxtral": round(v_time / max(w_audio_s, 1e-9), 3),
    }

    with open(f"{args.out}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    # markdown
    worst = sorted(rows, key=lambda r: r["wer_voxtral"] - r["wer_whisper"], reverse=True)
    lines = [
        "# Whisper vs. Voxtral-realtime — Nederlandse WER (FLEURS-nl)\n",
        f"- Clips: **{summary['clips']}** ({summary['audio_seconds']}s audio)",
        f"- Whisper-model: **{args.whisper_model}** ({args.device}/{ctype})",
        f"- Voxtral: **realtime** (streaming, via vLLM)\n",
        "| Model | WER ↓ | RTF ↓ |",
        "|---|---|---|",
        f"| Whisper {args.whisper_model} | {fmt_pct(corp_w)} | {summary['rtf_whisper']} |",
        f"| Voxtral-realtime | {fmt_pct(corp_v)} | {summary['rtf_voxtral']} |\n",
        "_Lager is beter. WER = Word Error Rate; RTF = Real-Time Factor (verwerkingstijd ÷ audioduur)._\n",
        "### Grootste verschillen (Voxtral t.o.v. Whisper)\n",
    ]
    for r in worst[:5]:
        lines += [
            f"**clip {r['i']} ({r['dur']}s)** — W={fmt_pct(r['wer_whisper'])} V={fmt_pct(r['wer_voxtral'])}",
            f"- ref:     {r['ref']}",
            f"- whisper: {r['whisper']}",
            f"- voxtral: {r['voxtral']}\n",
        ]
    with open(f"{args.out}.md", "w") as f:
        f.write("\n".join(lines))

    print("\n=== samenvatting ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\ngeschreven: {args.out}.json, {args.out}.md")


if __name__ == "__main__":
    main()
