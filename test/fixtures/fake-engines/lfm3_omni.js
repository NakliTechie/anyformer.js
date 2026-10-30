// A stand-in for diskformer's engines/lfm3_omni.js: records load and generate options; generate streams two pieces.
export const calls = [];
export class Lfm3Omni {
  static async load(opts) {
    calls.push({ type: 'load', opts });
    opts.onProgress?.({ status: 'download', file: 'onnx/decoder_q4f16.onnx', loaded: 1, total: 2 });
    return new Lfm3Omni();
  }
  async generate(messages, opts) {
    calls.push({ type: 'generate', messages, opts });
    // long: a reply that runs until its signal aborts, one piece per tick (the early-stop test)
    if (messages[0]?.content === 'long') {
      let n = 0;
      while (!opts.signal?.aborted && n < 1000) { opts.onText?.(`${n++} `); await new Promise((r) => setTimeout(r, 1)); }
      calls.push({ type: 'end', aborted: opts.signal?.aborted ?? false, pieces: n });
      return { text: '', stopReason: 'aborted' };
    }
    for (const piece of ['The capital', ' is Paris.']) {
      opts.onToken?.(1); opts.onText?.(piece);
      if (opts.output_modalities?.includes('audio')) opts.onAudio?.(new Float32Array(4));
    }
    return { text: 'The capital is Paris.', stopReason: 'eos' };
  }
  speech = { sampling_rate: 24000 };
  async dispose() { calls.push({ type: 'dispose' }); }
}
