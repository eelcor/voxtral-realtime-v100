"""Auto-imported via PYTHONPATH. Scales the realtime transcription delay.

Voxtral's streaming delay (default 6 tokens ≈ 480 ms) is the field
`AudioConfig.transcription_delay_ms`; `num_delay_tokens` derives from it. We wrap
`__post_init__` to multiply it by $VOXTRAL_DELAY_SCALE before the (multiple-of-frame)
asserts run, so scales 1.0/1.5/2.0/2.5 give 6/9/12/15 delay tokens. The resulting
values are dumped to /tmp/voxtral_delay.json so the sweep can label its points.
"""
import os


def _install():
    try:
        scale = float(os.environ.get("VOXTRAL_DELAY_SCALE", "1") or "1")
    except ValueError:
        scale = 1.0
    try:
        from mistral_common.tokens.tokenizers import audio as _a
    except Exception:
        return

    AC = _a.AudioConfig
    if getattr(AC, "_delay_patched", False):
        return
    orig = AC.__post_init__

    def patched(self):
        td = getattr(self, "transcription_delay_ms", None)
        if scale != 1.0 and td:
            self.transcription_delay_ms = td * scale
        orig(self)
        try:
            if getattr(self, "transcription_delay_ms", None) and self.is_streaming:
                import json
                with open("/tmp/voxtral_delay.json", "w") as f:
                    json.dump({
                        "scale": scale,
                        "transcription_delay_ms": self.transcription_delay_ms,
                        "num_delay_tokens": self.get_num_delay_tokens(),
                        "frame_rate": self.frame_rate,
                        "look_ahead_ms": self.streaming_look_ahead_ms,
                    }, f)
        except Exception:
            pass

    AC.__post_init__ = patched
    AC._delay_patched = True


_install()
