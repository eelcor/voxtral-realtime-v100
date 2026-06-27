#!/usr/bin/env python3
"""Plot WER vs. streaming delay from the sweep results into a PNG for the README."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/home/eelcor/voxtral_test/bench/delay_results"
META = "/tmp/sweep_clips/meta.json"
OUT = "/home/eelcor/voxtral-realtime-v100/bench/delay_vs_wer.png"

pts = []
for f in sorted(glob.glob(os.path.join(RESULTS, "delay_*.json"))):
    d = json.load(open(f))
    pts.append((d["num_delay_tokens"], d["delay_ms"], 100 * d["wer_voxtral"]))
pts.sort()
toks = [p[0] for p in pts]
ms = [p[1] for p in pts]
wv = [p[2] for p in pts]

meta = json.load(open(META))
whisper_wer = 100 * meta["whisper_wer"]
whisper_model = meta["whisper_model"]
n = len(meta["clips"])

plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#888"})
fig, ax = plt.subplots(figsize=(8, 5))

ax.axhline(whisper_wer, ls="--", lw=1.6, color="#6b7280",
           label=f"Whisper {whisper_model} (offline): {whisper_wer:.1f}%")
ax.plot(toks, wv, "-o", lw=2.4, ms=9, color="#a855f7", label="Voxtral-realtime (streaming)")
for t, m, w in pts:
    ax.annotate(f"{w:.1f}%", (t, w), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10, color="#a855f7", fontweight="bold")

# markeer het standaard-werkpunt (6 tokens)
if 6 in toks:
    i = toks.index(6)
    ax.annotate("standaard\n480 ms", (6, wv[i]), textcoords="offset points",
                xytext=(-2, -34), ha="center", fontsize=9, color="#555")

ax.set_xlabel("Streaming-delay (tokens)")
ax.set_ylabel("Word Error Rate (%)  ↓ beter")
ax.set_title(f"Voxtral-realtime: WER vs. latency  ·  FLEURS-nl, {n} clips")
ax.set_xticks(toks)
sec = ax.secondary_xaxis("top", functions=(lambda x: x * 80.0, lambda x: x / 80.0))
sec.set_xlabel("≈ latency (ms)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="best", frameon=False)
lo = min(min(wv), whisper_wer) - 1.0
hi = max(max(wv), whisper_wer) + 2.0
ax.set_ylim(max(0, lo), hi)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("geschreven:", OUT)
print("punten:", pts, "| whisper:", round(whisper_wer, 1))
