#!/usr/bin/env bash
# Voxtral-Mini-4B-Realtime via vLLM op EEN Tesla V100 (SM70 / Volta).
# Combineert het officiele realtime-recept (PIECEWISE cudagraphs) met het
# V100-recept (FLASH_ATTN_V100 backend, expandable_segments allocator).
#
# Vereist een vLLM-build met de FlashAttnV100Backend + de patch uit
# patches/whisper_causal_v100.patch (zie README).
#
# Realtime WebSocket: ws://<host>:<port>/v1/realtime  (auto-mount bij dit model)

set -euo pipefail

# Pad naar de V100-vLLM venv (met de whisper_causal.py-patch toegepast).
VENV=${VLLM_VENV:-$HOME/vllm-v100}
MODEL=${VLLM_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}
PORT=${VLLM_PORT:-8045}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}      # 4B past op 1 V100
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_DISABLE_COMPILE_CACHE=1

# Bewezen-werkende defaults op een 16GB V100 (single user). Block-pooling maakt
# KV duur (~0.7 MB/token), dus context bescheiden houden.
GPU_UTIL=${VLLM_GPU_MEM_UTIL:-0.92}
TP_SIZE=${VLLM_TP_SIZE:-1}
MAX_CTX=${VLLM_MAX_CTX:-4096}
MAX_BATCHED=${VLLM_MAX_BATCHED:-4096}
MAX_SEQS=${VLLM_MAX_SEQS:-1}

# Vanuit /tmp starten zodat de wheels (niet een source-tree) geladen worden.
cd /tmp

echo "Start Voxtral-realtime vLLM ($MODEL) op poort $PORT, GPU $CUDA_VISIBLE_DEVICES, TP=$TP_SIZE ..."

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name voxtral-realtime \
    --attention-backend FLASH_ATTN_V100 \
    --dtype float16 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_CTX" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --max-num-seqs "$MAX_SEQS" \
    --compilation_config '{"cudagraph_mode": "PIECEWISE"}' \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code
