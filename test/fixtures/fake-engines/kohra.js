// A stand-in for kohra.js: records from_pretrained options and the prompt generate gets.
export const calls = [];
export class DiffusionLM {
  static async from_pretrained(opts) {
    calls.push({ type: 'load', opts });
    opts.progress_callback?.({ status: 'progress', file: 'model_kv_q4f16.onnx_data', loaded: 1, total: 2, progress: 50 });
    return { async generate(prompt, o) { calls.push({ type: 'generate', prompt, o }); return { text: '391', tokens: 2, tokensPerSecond: 10 }; }, session: null };
  }
}
