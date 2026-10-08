// transformers.js: autoregressive ONNX text generation on WebGPU. The library loads from the CDN at load() time, so
// importing anyformer costs nothing until this back end is chosen.

const VERSION = '4.3.1';
const CDN = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${VERSION}/+esm`;

// dtype → the ONNX file transformers.js reads for it (decoder-only repos keep one graph per dtype under onnx/).
const FILES = { q4f16: 'onnx/model_q4f16.onnx', q4: 'onnx/model_q4.onnx', fp16: 'onnx/model_fp16.onnx', fp32: 'onnx/model.onnx' };

// f16 activations come last: on Apple Metal (Chrome 154, transformers.js 4.3.1) Qwen2.5-0.5B's q4f16 graph decodes
// to word salad and fp16 loops, while q4 (f32 activations) answers correctly (test/browser/transformers.html,
// 2026-10-08). kohra hit the same fp16 overflow on WebGPU (kohra README, "Known constraints").
function pickDtype(model, caps) {
  const have = new Set(model.files.onnx.map((f) => f.name));
  const order = caps.f16 ? ['q4', 'fp32', 'q4f16', 'fp16'] : ['q4', 'fp32'];
  return order.find((d) => have.has(FILES[d])) || null;
}

// The graph plus its external-data files (<graph>.data, <graph>_data, <graph>_data_1 …).
export const graphBytes = (model, file) =>
  model.files.all.filter((f) => f.name === file || f.name.startsWith(`${file}.data`) || f.name.startsWith(`${file}_data`))
    .reduce((n, f) => n + (f.size || 0), 0);

export const transformersBackend = {
  id: 'transformers',

  canRun(model, caps) {
    if (!model.formats.includes('onnx')) return { ok: false, reason: 'no ONNX graph in the repo' };
    // kohra's diffusion graphs are ONNX too, but transformers.js generate() is autoregressive only.
    if (model.tags?.includes('text-diffusion')) return { ok: false, reason: 'a text-diffusion model; transformers.js generate() is autoregressive only' };
    if (model.task !== 'text-generation') return { ok: false, reason: `task ${model.task || 'unknown'}; this back end runs text-generation only` };
    if (!caps.webgpu) return { ok: false, reason: 'no WebGPU adapter' };
    const dtype = pickDtype(model, caps);
    if (!dtype) return { ok: false, reason: `no ONNX graph at ${Object.values(FILES).join(', ')}` };
    const bytes = graphBytes(model, FILES[dtype]);
    return { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } };
  },

  async load(model, { caps, onProgress, signal }) {
    const dtype = pickDtype(model, caps);
    const { pipeline, TextStreamer, InterruptableStoppingCriteria } = await import(CDN);
    const gen = await pipeline('text-generation', model.repo, {
      device: 'webgpu', dtype, revision: model.revision || 'main',
      progress_callback: onProgress && ((p) => onProgress({ backend: 'transformers', ...p })),
    });
    signal?.throwIfAborted();
    return {
      async *generate(messages, { maxNewTokens = 256, signal } = {}) {
        // TextStreamer is callback-based; a queue turns it into the contract's async iterable.
        const queue = [];
        let wake = () => {}, done = false, failed = null, text = '', tokens = 0;
        const streamer = new TextStreamer(gen.tokenizer, {
          skip_prompt: true, skip_special_tokens: true,
          callback_function: (t) => { text += t; queue.push({ text, tokens }); wake(); },
          token_callback_function: () => { tokens++; },
        });
        const stop = new InterruptableStoppingCriteria();
        signal?.addEventListener('abort', () => stop.interrupt(), { once: true });
        gen(messages, { max_new_tokens: maxNewTokens, do_sample: false, streamer, stopping_criteria: stop })
          .catch((e) => { failed = e; })
          .finally(() => { done = true; wake(); });
        while (true) {
          if (queue.length) { yield queue.shift(); continue; }
          if (done) break;
          await new Promise((r) => { wake = r; });
        }
        if (failed) throw failed;
      },
      dispose: () => gen.dispose?.(),
    };
  },
};
