#!/usr/bin/env bash
# Past de V100-patch toe op een vLLM-installatie zodat het Voxtral-realtime
# audio-encoder (block-pooling attention) de Triton-based FlashAttnV100Backend
# accepteert. Installeert ook de audio-dependency die vLLM bij profilering nodig heeft.
#
# Gebruik:
#   VLLM_VENV=$HOME/vllm-v100 ./apply_patch.sh
#
# Vereist een vLLM-build die de FlashAttnV100Backend levert (sm_70/Volta).

set -euo pipefail

VENV=${VLLM_VENV:-$HOME/vllm-v100}
PATCH="$(cd "$(dirname "$0")" && pwd)/whisper_causal_v100.patch"

PKG_DIR="$("$VENV/bin/python" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
TARGET="$PKG_DIR/model_executor/models/whisper_causal.py"

echo "vLLM-package : $PKG_DIR"
echo "doelbestand  : $TARGET"

[ -f "$TARGET" ] || { echo "FOUT: $TARGET niet gevonden"; exit 1; }
[ -f "$TARGET.orig" ] || cp "$TARGET" "$TARGET.orig"

# idempotent: detecteer of de patch al is toegepast
if grep -q "TritonAttentionBackend" "$TARGET" && grep -q "_bp_allowed" "$TARGET"; then
    echo "Patch lijkt al toegepast; sla over."
else
    # patch toepassen vanuit de site-packages root (paden in de diff zijn vllm/...)
    ( cd "$PKG_DIR/.." && patch -p1 < "$PATCH" )
    echo "Patch toegepast (backup op $TARGET.orig)."
fi

echo "Audio-dependency installeren ..."
"$VENV/bin/pip" install -q soundfile librosa soxr

echo "Klaar. Start de server met serve_voxtral_vllm.sh"
