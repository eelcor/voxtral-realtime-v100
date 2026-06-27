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
n = len(meta["clips"])

# Whisper-baselines: v3 uit meta + extra whisper_<model>.json bestanden
baselines = {meta["whisper_model"]: 100 * meta["whisper_wer"]}
for f in glob.glob(os.path.join(os.path.dirname(META), "whisper_*.json")):
    d = json.load(open(f))
    baselines[d["whisper_model"]] = 100 * d["whisper_wer"]
all_wer = wv + list(baselines.values())

plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#888"})
fig, ax = plt.subplots(figsize=(8, 5))

bl_colors = {"large-v3": "#374151", "large-v2": "#9ca3af", "medium": "#cbd5e1"}
for name, w in sorted(baselines.items(), key=lambda kv: kv[1]):
    ax.axhline(w, ls="--", lw=1.6, color=bl_colors.get(name, "#6b7280"),
               label=f"Whisper {name} (offline): {w:.1f}%")
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
ax.set_ylim(max(0, min(all_wer) - 1.0), max(all_wer) + 2.0)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("geschreven:", OUT)
print("punten:", pts, "| baselines:", {k: round(v, 1) for k, v in baselines.items()})
