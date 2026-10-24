// A stand-in for diskformer's ggml/llama.js: records each call; files "download" into an in-memory list.
export const calls = [];
const stored = [];
export class Llama {
  async list() { return stored; }
  async download({ url, name, size }, onProgress) {
    calls.push({ type: 'download', url, name, size });
    onProgress?.({ copied: size, total: size });
    stored.push({ name, size });
  }
  async load(opts) { calls.push({ type: 'load', ...opts }); return { slots: 8 }; }
  abort() {}
  terminate() {}
}
