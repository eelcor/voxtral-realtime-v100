#!/usr/bin/env bash
# Klein LLM (Gemma 4 26B-A4B, MoE) via llama-server voor de webapp-AI-modi.
# Draait op de vrije GPU's; OpenAI-compatibel op poort 8052.
set -euo pipefail
LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}
: "${GEMMA_MODEL:?Zet GEMMA_MODEL naar het pad van het gemma-4 GGUF-bestand}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
exec "$LLAMA_BIN" --model "$GEMMA_MODEL" \
  --host 127.0.0.1 --port "${LLM_PORT:-8052}" \
  --ctx-size 16384 --n-gpu-layers 99 \
  --split-mode layer --flash-attn on --parallel 2 \
  --threads 20 --jinja
