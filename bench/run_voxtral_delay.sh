#!/usr/bin/env bash
# Start de Voxtral-realtime server op poort 8045/GPU2 met een geschaalde
# streaming-delay. VOXTRAL_DELAY_SCALE=1.0/1.5/2.0/2.5 -> 6/9/12/15 delay tokens.
export CUDA_VISIBLE_DEVICES=2
export VLLM_MAX_CTX=3072
export VLLM_MAX_SEQS=4
export VLLM_GPU_MEM_UTIL=0.92
export VLLM_PORT=8045
export PYTHONPATH=/home/eelcor/voxtral_test/bench/delay_patch:${PYTHONPATH:-}
export VOXTRAL_DELAY_SCALE=${VOXTRAL_DELAY_SCALE:-1.0}
exec bash /home/eelcor/voxtral_test/serve_voxtral_vllm.sh
