#!/usr/bin/env bash
# Start de Voxtral realtime-transcriptie-webapp over HTTPS (self-signed cert,
# nodig omdat browsers de microfoon alleen in een secure context vrijgeven).
#
# Env:
#   WEBAPP_PORT   (default 8443)
#   VOXTRAL_WS    (default ws://127.0.0.1:8045/v1/realtime)
#   VLLM_VENV     (default /home/eelcor/vllm-v100)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=${VLLM_VENV:-/home/eelcor/vllm-v100}
PORT=${WEBAPP_PORT:-8443}
CERT="$HERE/cert.pem"; KEY="$HERE/key.pem"

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
  echo "Self-signed cert genereren ..."
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" \
    -days 825 -subj "/CN=voxtral-realtime" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null
fi

export VOXTRAL_WS=${VOXTRAL_WS:-ws://127.0.0.1:8045/v1/realtime}
export VOXTRAL_HEALTH=${VOXTRAL_HEALTH:-http://127.0.0.1:8045/health}
export VOXTRAL_MODEL=${VOXTRAL_MODEL:-voxtral-realtime}

echo "Webapp op https://0.0.0.0:$PORT  (Voxtral: $VOXTRAL_WS)"
cd "$HERE"
exec "$VENV/bin/python" -m uvicorn server:app \
  --host 0.0.0.0 --port "$PORT" \
  --ssl-keyfile "$KEY" --ssl-certfile "$CERT" \
  --log-level warning
