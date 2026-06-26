# Voxtral realtime transcription on Tesla V100

Run **Mistral's `Voxtral-Mini-4B-Realtime-2602`** streaming speech-to-text on a single
**Tesla V100 (Volta, sm_70)** via vLLM — keeping up with real time and then some.

> Measured: **RTF ≈ 0.41** (62.5 s of audio transcribed in 25.7 s — ~2.4× faster than
> real time), first-token latency ~0.1–0.3 s, accurate punctuated output, Dutch and 12
> other languages, on one 16 GB V100.

The realtime model does **not** run on a V100 out of the box: its audio encoder uses a
block-pooling attention that hard-requires Ampere+ FlashAttention. This repo contains a
small two-line vLLM patch that makes it work on Volta, plus ready-to-use server and client
scripts (including a live microphone demo).

## Why this is non-trivial

| Approach | Result on a single V100 |
|---|---|
| 🤗 transformers (`VoxtralRealtimeForConditionalGeneration`) | Works, **but RTF ≈ 1.18** — falls ~20 % behind, lag accumulates |
| `torch.compile` on the transformers model | Fails (`NameError` from Inductor codegen on this custom model) |
| vLLM, unpatched | Engine fails to start: `NotImplementedError: FlashAttnV100Backend is not yet supported` |
| **vLLM + this patch** | ✅ **RTF ≈ 0.41** — keeps up real time with headroom |

The bottleneck was never raw FLOPs — a V100 has plenty for a 4B model. It was kernel-launch
overhead in the eager transformers loop. vLLM's CUDA graphs remove exactly that.

## Requirements

- **A Tesla V100** (Volta, sm_70), 16 GB or 32 GB. One card is enough — the model is 4B
  and fits in ~10 GB. More V100s let you run more concurrent instances; they are **not**
  required, and tensor-parallel with this patch is untested.
- **The 1Cat-vLLM Volta fork** — [github.com/1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).
  This is the hard prerequisite: it's a public V100/sm_70 vLLM fork that ships the
  Triton-based `FlashAttnV100Backend` (`--attention-backend FLASH_ATTN_V100`) and is built
  against CUDA 12.8 (stock vLLM has no Volta backend, and default cu13x wheels drop Volta).
  Install its prebuilt wheels into a Python 3.12 venv, e.g.:
  ```bash
  python3.12 -m venv ~/vllm-v100 && source ~/vllm-v100/bin/activate
  pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu128 \
    https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl
  ```
  This repo's patch was developed against **1Cat-vLLM v1.0.0 (vLLM 1.0.0)**; on other
  versions the diff context may need a tweak.
- **The model weights**: `mistralai/Voxtral-Mini-4B-Realtime-2602` (auto-downloaded on first run).

> Tested on a 16 GB V100 with the defaults below. On a 32 GB V100 you can raise
> `--max-model-len` and `--max-num-seqs` substantially.

## Quick start

```bash
# 0) Install 1Cat-vLLM into ~/vllm-v100 (see Requirements), then apply the patch
#    (idempotent; backs up the original whisper_causal.py).
VLLM_VENV=$HOME/vllm-v100 ./patches/apply_patch.sh

# 1) Start the server (single V100, port 8045). Wait for "Application startup complete."
VLLM_VENV=$HOME/vllm-v100 ./serve_voxtral_vllm.sh

# 2a) Transcribe a 16 kHz WAV file
$HOME/vllm-v100/bin/python clients/test_realtime_ws.py sample.wav

# 2b) Live microphone -> transcription (PipeWire default source)
$HOME/vllm-v100/bin/python clients/mic_realtime_vllm.py
#     specific source:  --source <node.name>     (see `wpctl status`)
#     ALSA instead:     --backend arecord --source plughw:CARD=PCH,DEV=0
```

Language is auto-detected — just talk.

## How the realtime API works

The realtime model is served over a **WebSocket** at `ws://<host>:8045/v1/realtime`
(auto-mounted by vLLM when the model supports the `realtime` task). The HTTP
`/v1/audio/transcriptions` endpoint does **not** work for this model — it requires a
multimodal embedding at every step, which only the streaming protocol provides.

Message flow (OpenAI-realtime style):

```
client → {"type":"session.update","model":"voxtral-realtime"}
client → {"type":"input_audio_buffer.append","audio":"<base64 PCM16 @16kHz>"}   (repeat)
client → {"type":"input_audio_buffer.commit"}                 # starts generation
server → {"type":"transcription.delta","delta":"..."}         (streamed)
client → {"type":"input_audio_buffer.commit","final":true}    # end of input
server → {"type":"transcription.done","text":"..."}
```

See `clients/test_realtime_ws.py` for a complete, minimal client.

**Latency knob:** Voxtral's streaming delay is configurable (multiples of 80 ms, 80–1200 ms
plus 2400 ms) via `num_delay_tokens` in the model's `tekken.json`. Default 6 tokens ≈ 480 ms,
which matches offline accuracy. Lower = snappier, slightly less accurate.

## The patch

`patches/whisper_causal_v100.patch` changes two things in
`vllm/model_executor/models/whisper_causal.py`:

1. **Relax the backend gate.** `create_whisper_attention_backend_with_block_pooling`
   only accepted `FlashAttentionBackend` subclasses (Ampere+). We also accept
   `TritonAttentionBackend`, which the V100 backend (`FlashAttnV100Backend`) subclasses.
2. **Make the KV-cache shape backend-agnostic.** The block-pooling override hardcoded
   FlashAttention's `(2, num_blocks, …)` layout. We delegate to the underlying backend's
   `get_kv_cache_shape(...)` (with stretched `block_size` and reduced `num_kv_heads`), so it
   is correct for both FlashAttention `(2, blocks, …)` and Triton `(blocks, 2, …)`.

Everything else the wrapper needs (`do_kv_cache_update`, the `forward` signature,
`forward_includes_kv_cache_update`) already matches the Triton backend, so no further
changes are required.

> Memory note: block-pooling makes the KV cache expensive (~0.7 MB/token), so keep
> `--max-model-len` modest. The provided defaults (`max-model-len 4096`, `max-num-seqs 1`,
> `gpu-memory-utilization 0.92`) fit comfortably on a 16 GB V100 for single-user use.

## Repo layout

```
serve_voxtral_vllm.sh             vLLM launch script (single V100, port 8045)
patches/
  whisper_causal_v100.patch       the two-line fix (unified diff)
  apply_patch.sh                  idempotent applier + audio deps
clients/
  mic_realtime_vllm.py            live microphone -> websocket (the main demo)
  test_realtime_ws.py             WAV file -> websocket, prints transcript + RTF
transformers_baseline/            reference path (works but lags) + benchmarks
  transcribe_live.py              mic -> transformers streaming (with a --meter mode)
  test_offline.py                 validate transcription on a reference clip
  bench_rtf.py                    measure real-time factor (shows the ~1.18 lag)
  bench_compile.py                torch.compile experiment (documents why it fails)
requirements-transformers.txt     deps for the transformers_baseline scripts only
```

## Transformers baseline (reference / why vLLM)

The `transformers_baseline/` scripts are the simpler, dependency-light path. They produce
identical transcription quality but **lag ~20 % behind real time** on a V100 — useful as a
reference and to reproduce the diagnosis (`bench_rtf.py`). They need their own venv; see
`requirements-transformers.txt` (note the CUDA-12 torch pin for Volta).

## Credits & license

- Model: [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
  by Mistral AI (Apache-2.0).
- Serving: [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0). The patch is a
  derivative of vLLM source.
- V100/Volta support: the [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) fork,
  which provides the `FlashAttnV100Backend` this work builds on. Huge thanks — without it
  none of this runs on Volta.
- This repo is released under the **Apache-2.0** license (see `LICENSE`).
