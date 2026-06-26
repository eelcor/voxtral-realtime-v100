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

# Detecteer of de fix er al is:
#  - upstream (v1.2.1+): de gate gebruikt _SUPPORTED_BACKENDS, of
#  - get_kv_cache_shape delegeert al naar de onderliggende backend, of
#  - deze patch is al toegepast (_bp_allowed-marker).
if grep -q "_SUPPORTED_BACKENDS" "$TARGET" \
   || grep -q "_bp_allowed" "$TARGET" \
   || grep -q "underlying_attn_backend.get_kv_cache_shape" "$TARGET"; then
    echo "De fix is al aanwezig (upstream v1.2.1+ of eerder gepatcht) — geen patch nodig."
else
    [ -f "$TARGET.orig" ] || cp "$TARGET" "$TARGET.orig"
    # eerst droog draaien zodat een versie-mismatch niet half patcht
    if ( cd "$PKG_DIR/.." && patch -p1 --dry-run < "$PATCH" ) >/dev/null 2>&1; then
        ( cd "$PKG_DIR/.." && patch -p1 < "$PATCH" )
        echo "Patch toegepast (backup op $TARGET.orig)."
    else
        echo "WAARSCHUWING: de patch past niet schoon op deze vLLM-versie."
        echo "  - Op v1.2.1+ is de fix al upstream en is de patch niet nodig."
        echo "  - Op een andere versie: poort de twee wijzigingen handmatig"
        echo "    (zie README, sectie 'The patch')."
        exit 1
    fi
fi

echo "Audio-dependency installeren ..."
"$VENV/bin/pip" install -q soundfile librosa soxr

echo "Klaar. Start de server met serve_voxtral_vllm.sh"
