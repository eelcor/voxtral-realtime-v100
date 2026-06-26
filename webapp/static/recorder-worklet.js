// Downsamplet de microfoon-audio (meestal 48kHz) naar 16kHz mono PCM16
// en post elke ~100ms een Int16-frame naar de main thread.
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate; // sampleRate = context-rate (worklet global)
    this.frame = 1600;       // 100 ms @ 16kHz
    this._in = [];           // ruwe input-samples (Float32)
    this._out = [];          // geresamplede samples
    this._pos = 0;           // fractionele leespositie in _in
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) {
      for (let i = 0; i < ch.length; i++) this._in.push(ch[i]);
    }
    // lineaire resampling naar 16kHz
    while (this._pos + 1 < this._in.length) {
      const idx = Math.floor(this._pos);
      const frac = this._pos - idx;
      this._out.push(this._in[idx] * (1 - frac) + this._in[idx + 1] * frac);
      this._pos += this.ratio;
      if (this._out.length >= this.frame) {
        this._emit(this._out.splice(0, this.frame));
      }
    }
    const consumed = Math.floor(this._pos);
    if (consumed > 0) {
      this._in.splice(0, consumed);
      this._pos -= consumed;
    }
    return true;
  }
  _emit(samples) {
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      let v = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
  }
}
registerProcessor('recorder-processor', RecorderProcessor);
