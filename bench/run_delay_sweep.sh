#!/usr/bin/env bash
# Sweep de Voxtral streaming-delay (6/9/12/15 tokens) en scoor WER per stap.
# Herstelt aan het eind de productie-server (delay 6) detached op 8045/GPU2.
set -uo pipefail
BENCH=/home/eelcor/voxtral_test/bench
PY=/home/eelcor/whisper-bench/bin/python
OUTDIR=$BENCH/delay_results
mkdir -p "$OUTDIR"

stop_voxtral() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for p in $(pgrep -f "VLLM::EngineCore"); do kill -9 "$p" 2>/dev/null; done
  for i in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '3p')
    [ "$u" -lt 3000 ] && break; sleep 2
  done
}

wait_ready() {
  for i in $(seq 1 70); do
    [ "$(curl -s -m3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8045/v1/models 2>/dev/null)" = "200" ] && return 0
    sleep 3
  done
  return 1
}

for scale in 1.0 1.5 2.0 2.5; do
  echo "===== delay scale $scale ====="
  stop_voxtral
  rm -f /tmp/voxtral_delay.json
  VOXTRAL_DELAY_SCALE=$scale bash "$BENCH/run_voxtral_delay.sh" > /tmp/voxtral_delay_server.log 2>&1 &
  if ! wait_ready; then echo "server kwam niet op voor scale $scale"; tail -5 /tmp/voxtral_delay_server.log; continue; fi
  sleep 2
  ( cd "$BENCH" && $PY sweep_voxtral.py --out "$OUTDIR/delay_${scale}.json" )
done

echo "===== sweep klaar, productie-server (delay 6) herstellen ====="
stop_voxtral
setsid bash /home/eelcor/voxtral_test/webapp/run_voxtral_8045.sh > /tmp/voxtral_restore.log 2>&1 < /dev/null &
wait_ready && echo "productie-Voxtral terug op 8045" || echo "LET OP: herstart productie-Voxtral handmatig"
echo "ALLE DELAY-RUNS KLAAR"
