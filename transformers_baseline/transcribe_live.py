#!/usr/bin/env python3
"""
Live realtime-transcriptie met Voxtral-Mini-4B-Realtime op een Tesla V100.

Audiobron via PipeWire (pw-record, default) of ALSA (arecord). Standaard de
NanoKVMPro-stereobron. Audio wordt als kale PCM (16kHz mono s16) ingelezen en
in overlappende vensters aan de streaming-generator gevoerd; tekst verschijnt
incrementeel.

Latency/kwaliteit-knop --delay-tokens: 1=~80ms, 6=~480ms (default), 30=~2.4s.

Modi:
  --info    model laden, config/VRAM tonen, stoppen (geen mic)
  --meter   alleen live ingangsniveau tonen (geen model) — om de bron te checken
  (geen)    live transcriptie

Stoppen: Ctrl+C.
"""
import argparse
import subprocess
import sys
import threading
import time

import numpy as np
import torch

import os

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
# None => PipeWire default-bron. Zet VOXTRAL_PW_SOURCE of geef --source <node.name>.
DEFAULT_PW_SOURCE = os.environ.get("VOXTRAL_PW_SOURCE")
SR = 16000


def parse_args():
    p = argparse.ArgumentParser(description="Live Voxtral realtime transcriptie")
    p.add_argument("--device", default="cuda:0", help="CUDA device (default cuda:0)")
    p.add_argument("--backend", choices=["pipewire", "arecord"], default="pipewire",
                   help="Audio-backend (default pipewire)")
    p.add_argument("--source", default=None,
                   help="pipewire: node-naam/serial (default NanoKVMPro). "
                        "arecord: ALSA-device (bv. plughw:CARD=PCH,DEV=0)")
    p.add_argument("--delay-tokens", type=int, default=None,
                   help="Streaming-delay in tokens (1=80ms .. 30=2.4s). Default = config (6).")
    p.add_argument("--info", action="store_true", help="Model+config tonen en stoppen.")
    p.add_argument("--meter", action="store_true",
                   help="Alleen live ingangsniveau tonen (geen model).")
    return p.parse_args()


def capture_cmd(backend, source):
    """Bouw het opname-commando; beide leveren kale s16 mono 16kHz PCM op stdout."""
    if backend == "pipewire":
        tgt = source or DEFAULT_PW_SOURCE
        cmd = ["pw-record", "--rate", str(SR), "--channels", "1", "--format", "s16"]
        if tgt:
            cmd += ["--target", tgt]
        return cmd + ["-"]
    else:
        dev = source or "default"
        return ["arecord", "-D", dev, "-f", "S16_LE", "-r", str(SR),
                "-c", "1", "-t", "raw", "-q", "-"]


class MicBuffer:
    """Groeiende, thread-safe audiobuffer gevoed door een opname-subprocess."""
    def __init__(self, cmd):
        self.lock = threading.Condition()
        self.samples = np.zeros(0, dtype=np.float32)
        self.stopped = False
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _read_loop(self):
        block = 1600 * 2  # 0.1s aan int16-bytes
        try:
            while not self.stopped:
                data = self.proc.stdout.read(block)
                if not data:
                    break
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                with self.lock:
                    self.samples = np.concatenate([self.samples, arr])
                    self.lock.notify_all()
        finally:
            with self.lock:
                self.stopped = True
                self.lock.notify_all()

    def wait_for(self, end_idx):
        with self.lock:
            while len(self.samples) < end_idx and not self.stopped:
                self.lock.wait()
            return len(self.samples) >= end_idx

    def slice(self, start, end):
        with self.lock:
            return self.samples[start:end].copy()

    def level(self):
        with self.lock:
            n = len(self.samples)
        return n

    def stop(self):
        self.stopped = True
        try:
            self.proc.terminate()
        except Exception:
            pass


def run_meter(cmd):
    """Toon ~10x/s het RMS/piek-niveau van de binnenkomende audio."""
    print(f"[meter] {' '.join(cmd)}", file=sys.stderr)
    print("[meter] Ctrl+C om te stoppen. Praat/route audio en kijk of de balk uitslaat.\n",
          file=sys.stderr, flush=True)
    mic = MicBuffer(cmd)
    last = 0
    try:
        while not mic.stopped:
            time.sleep(0.1)
            n = mic.level()
            if n <= last:
                continue
            recent = mic.slice(max(0, n - 1600), n)  # laatste ~0.1s
            last = n
            if len(recent) == 0:
                continue
            rms = float(np.sqrt(np.mean(recent**2)))
            peak = float(np.max(np.abs(recent)))
            bars = int(min(1.0, rms * 8) * 40)
            print(f"\rRMS {rms:.4f}  piek {peak:.3f}  |{'#'*bars:<40}|",
                  end="", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        print("\n[meter] gestopt.", file=sys.stderr)


def load(device):
    from transformers import (VoxtralRealtimeForConditionalGeneration,
                              VoxtralRealtimeProcessor)
    print(f"[laden] {MODEL_ID} op {device} (fp16, sdpa) ...", file=sys.stderr, flush=True)
    processor = VoxtralRealtimeProcessor.from_pretrained(MODEL_ID)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=device,
        attn_implementation="sdpa").eval()
    return processor, model


def describe(processor, device):
    hop = processor.feature_extractor.hop_length
    apt = processor.audio_length_per_tok
    ms = apt * hop / SR * 1000.0
    ndt = processor.num_delay_tokens
    print("=== Voxtral realtime config ===", file=sys.stderr)
    print(f"  ms per token      : {ms:.0f} ms", file=sys.stderr)
    print(f"  num_delay_tokens  : {ndt}  ->  ~{ndt*ms:.0f} ms delay", file=sys.stderr)
    print(f"  eerste chunk      : {processor.num_samples_first_audio_chunk} samples "
          f"({processor.num_samples_first_audio_chunk/SR*1000:.0f} ms)", file=sys.stderr)
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        print(f"  VRAM              : {torch.cuda.memory_allocated(idx)/1e9:.2f} GB", file=sys.stderr)
    print("===============================", file=sys.stderr, flush=True)


def main():
    args = parse_args()
    cmd = capture_cmd(args.backend, args.source)

    if args.meter:
        run_meter(cmd)
        return

    from transformers import TextIteratorStreamer
    processor, model = load(args.device)
    if args.delay_tokens is not None:
        try:
            processor.mistral_common_audio_config.num_delay_tokens = args.delay_tokens
            print(f"[knop] num_delay_tokens = {args.delay_tokens}", file=sys.stderr)
        except Exception as e:
            print(f"[waarschuwing] kon delay niet zetten: {e}", file=sys.stderr)
    describe(processor, args.device)
    if args.info:
        return

    hop = processor.feature_extractor.hop_length
    win = processor.feature_extractor.win_length
    dev = args.device
    n_per = processor.num_samples_per_audio_chunk

    mic = MicBuffer(cmd)
    print(f"\n[bron] {' '.join(cmd)}", file=sys.stderr)
    print("[klaar] Transcriptie verschijnt live. Ctrl+C om te stoppen.\n",
          file=sys.stderr, flush=True)

    n_first = processor.num_samples_first_audio_chunk
    if not mic.wait_for(n_first):
        print("Geen audio ontvangen.", file=sys.stderr)
        mic.stop()
        return
    first_inputs = processor(mic.slice(0, n_first), is_streaming=True,
                             is_first_audio_chunk=True, return_tensors="pt")

    def feature_generator():
        yield first_inputs.input_features.to(dev, torch.float16)
        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        start_idx = mel_frame_idx * hop - win // 2
        while True:
            end_idx = start_idx + n_per
            if not mic.wait_for(end_idx):
                break
            inp = processor(mic.slice(start_idx, end_idx), is_streaming=True,
                            is_first_audio_chunk=False, return_tensors="pt")
            yield inp.input_features.to(dev, torch.float16)
            mel_frame_idx += processor.audio_length_per_tok
            start_idx = mel_frame_idx * hop - win // 2

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=first_inputs.input_ids.to(dev),
        input_features=feature_generator(),
        num_delay_tokens=first_inputs.num_delay_tokens,
        streamer=streamer, do_sample=False, max_new_tokens=1000000)
    # NIET-daemon: we sluiten 'm netjes af i.p.v. mid-CUDA teardown (voorkomt core dump).
    gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    gen_thread.start()

    try:
        for text in streamer:
            print(text, end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n[stoppen] bron afsluiten ...", file=sys.stderr, flush=True)
    finally:
        mic.stop()  # -> generator's wait_for geeft False -> model.generate eindigt netjes
        try:
            for _ in streamer:  # queue leegtrekken zodat generate niet blokkeert
                pass
        except Exception:
            pass
        gen_thread.join(timeout=10)


if __name__ == "__main__":
    main()
