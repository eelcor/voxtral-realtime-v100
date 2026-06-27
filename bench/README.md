# Benchmark: Whisper vs. Voxtral-realtime (Dutch WER)

Measures transcription quality of the realtime Voxtral server against
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) on Dutch read speech
([FLEURS-nl](https://huggingface.co/datasets/google/fleurs)), reported as **Word
Error Rate (WER)**.

## Setup

```bash
python3.12 -m venv ~/whisper-bench
~/whisper-bench/bin/pip install -r bench/requirements-bench.txt
```

The Voxtral realtime server must be running (see the repo README):

```bash
VLLM_VENV=$HOME/vllm-v100 ./serve_voxtral_vllm.sh      # port 8045
```

## Run

```bash
# WER over 80 FLEURS-nl clips, Whisper large-v3 on GPU 0
~/whisper-bench/bin/python bench/bench.py --n 80 --whisper-model large-v3 --gpu 0

# size-matched-ish reference
~/whisper-bench/bin/python bench/bench.py --n 80 --whisper-model medium --gpu 0

# CPU instead of GPU for Whisper (int8)
~/whisper-bench/bin/python bench/bench.py --n 40 --device cpu

# your own recording, side-by-side (no WER, no reference needed)
~/whisper-bench/bin/python bench/bench.py --audio my_recording.wav
```

Writes `results.json` (per-clip detail) and `results.md` (summary + the clips
where the two models disagree most).

## What it does

- Pulls `--n` clips from FLEURS-nl `test` (streaming; audio decoded with
  `soundfile`, so no `torchcodec`/torch needed).
- Transcribes each clip with **Whisper** (offline, full context) and with
  **Voxtral-realtime** (streaming, over the WebSocket — `voxtral_file.py`).
- Light normalization (lowercase, strip punctuation, collapse whitespace), then
  corpus WER via `jiwer`.

## Caveats — read before trusting the number

- **Not apples-to-apples.** Whisper here is *offline* and sees the whole clip;
  Voxtral-realtime is *streaming* and commits with limited right-context. This
  benchmarks "the quality you actually get live from Voxtral" vs "a strong offline
  reference", not two models on the same task.
- **FLEURS is clean read speech**, not noisy conversation — real-world
  conversational WER will be higher for both. Use `--audio` on your own recordings
  for a representative qualitative check.
- **WER is inflated by formatting**, not just real errors: digits vs. spelled-out
  numbers, compound-word splits (`continuümmodel` → `Continuum-model`), casing of
  names. The normalization is deliberately light.
- **Voxtral RTF reported here is not its streaming latency.** `bench.py` bursts each
  clip and waits out a drain timeout per clip, so the printed Voxtral RTF is
  dominated by per-clip overhead. For true streaming throughput see the repo's RTF
  notes (~0.4 on a single V100).

## `voxtral_file.py`

Standalone helper: transcribe any audio file through the realtime WebSocket.

```bash
~/whisper-bench/bin/python bench/voxtral_file.py my_clip.wav
```

It burst-feeds the file, appends trailing silence, and reads until the output
drains (using a plain `commit`, not `final`) so the last word — which otherwise
sits in the model's ~480 ms output-delay buffer — is not clipped.

## Latency vs. accuracy sweep

`run_delay_sweep.sh` measures WER at 6/9/12/15 streaming-delay tokens and plots it
(see the chart + table in the repo README). It works by scaling the model's
`transcription_delay_ms` at server start via `delay_patch/sitecustomize.py`
(auto-imported through `PYTHONPATH`, controlled by `$VOXTRAL_DELAY_SCALE`).

```bash
~/whisper-bench/bin/python bench/prep_clips.py --n 50 --gpu 0   # clips + large-v3 baseline (once)
~/whisper-bench/bin/python bench/whisper_eval.py --model large-v2 --dir /tmp/sweep_clips  # extra baseline line
bash bench/run_delay_sweep.sh                                   # stops/restarts the server per delay
~/whisper-bench/bin/python bench/plot_delay.py                  # -> bench/delay_vs_wer.png
```

`plot_delay.py` draws a dashed line for every `whisper_*.json` baseline in the clips dir, so add
as many Whisper models as you like with `whisper_eval.py`. On 50 clips the curve crosses **below
offline large-v2 (8.5 %)** by 9 delay tokens.

> The sweep **restarts the Voxtral server** for each delay (it reuses the single
> V100 slot), so live transcription on that server pauses while it runs. It restores
> the default-delay server (detached, port 8045) at the end.
