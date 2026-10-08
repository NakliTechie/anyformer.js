// transformers.js: ONNX models on WebGPU, text generation plus the other pipeline tasks below. The library loads from the CDN at load() time, so
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

// The non-generation tasks this back end runs, each through transformers.js's own pipeline(). They load the fp32
// graphs: the f16 paths gave wrong output on Metal for text generation, and these models are small.
const PIPELINES = {
  'feature-extraction': { options: { pooling: 'mean', normalize: true }, out: (t) => t.tolist() },
  'sentence-similarity': { task: 'feature-extraction', options: { pooling: 'mean', normalize: true }, out: (t) => t.tolist() },
  'automatic-speech-recognition': {},
  'object-detection': { image: true },
  'image-classification': { image: true },
  'depth-estimation': { image: true },
  'image-segmentation': { image: true },
  'token-classification': {},
  'text-classification': {},
};

// fp32 graphs: onnx/*.onnx without a precision suffix (model.onnx, encoder_model.onnx, decoder_model_merged.onnx).
const FP32 = /^onnx\/[a-z_]+(?<!_fp16|_q4|_q4f16|_quantized|_int8|_uint8|_bnb4|_q8)\.onnx$/;
const fp32Bytes = (model) => model.files.onnx.filter((f) => FP32.test(f.name)).reduce((n, f) => n + graphBytes(model, f.name), 0);

export const transformersBackend = {
  id: 'transformers',
  tasks: ['text-generation', ...Object.keys(PIPELINES)],

  canRun(model, caps, { task } = {}) {
    if (!model.formats.includes('onnx')) return { ok: false, code: 'format', reason: 'no ONNX graph in the repo' };
    // kohra's diffusion graphs are ONNX too, but transformers.js generate() is autoregressive only.
    if (model.tags?.includes('text-diffusion')) return { ok: false, code: 'model', reason: 'a text-diffusion model; transformers.js generate() is autoregressive only' };
    const want = task || model.task;
    if (!transformersBackend.tasks.includes(want)) return { ok: false, code: 'task', reason: `task ${want || 'unknown'}; this back end runs ${transformersBackend.tasks.join(', ')}` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (want !== 'text-generation') {
      const bytes = fp32Bytes(model);
      if (!bytes) return { ok: false, code: 'format', reason: 'no fp32 ONNX graph (onnx/model.onnx or encoder/decoder graphs)' };
      return { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } };
    }
    const dtype = pickDtype(model, caps);
    if (!dtype) return { ok: false, code: 'format', reason: `no ONNX graph at ${Object.values(FILES).join(', ')}` };
    const bytes = graphBytes(model, FILES[dtype]);
    return { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } };
  },

  async load(model, { task, caps, onProgress, signal }) {
    const want = task || model.task;
    const lib = await import(CDN);
    const progress_callback = onProgress && ((p) => onProgress({ backend: 'transformers', ...p }));
    if (want !== 'text-generation') return loadPipeline(lib, model, want, { progress_callback, signal });
    const dtype = pickDtype(model, caps);
    const { pipeline, TextStreamer, InterruptableStoppingCriteria } = lib;
    const gen = await pipeline('text-generation', model.repo, {
      device: 'webgpu', dtype, revision: model.revision || 'main', progress_callback,
    });
    signal?.throwIfAborted();
    return {
      task: 'text-generation',
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

// A non-generation task: one pipeline, run(input, opts) calls it. ImageData (from a canvas) becomes a RawImage; URLs,
// RawImage and, for speech, 16 kHz Float32Array samples pass through as transformers.js takes them.
async function loadPipeline(lib, model, task, { progress_callback, signal }) {
  const spec = PIPELINES[task];
  const pipe = await lib.pipeline(spec.task || task, model.repo, {
    device: 'webgpu', dtype: 'fp32', revision: model.revision || 'main', progress_callback,
  });
  signal?.throwIfAborted();
  const image = (x) => (typeof ImageData !== 'undefined' && x instanceof ImageData ? new lib.RawImage(x.data, x.width, x.height, 4) : x);
  return {
    task,
    async run(input, opts = {}) {
      const out = await pipe(spec.image ? image(input) : input, { ...spec.options, ...opts });
      return spec.out ? spec.out(out) : out;
    },
    dispose: () => pipe.dispose?.(),
  };
}
