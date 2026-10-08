// transformers.js: ONNX models on WebGPU, text generation plus the other pipeline tasks below. The library loads from the CDN at load() time, so
// importing anyformer costs nothing until this back end is chosen.


// dtype → the ONNX file transformers.js reads for it (decoder-only repos keep one graph per dtype under onnx/).
const FILES = {
  q4: 'onnx/model_q4.onnx', q2: 'onnx/model_q2.onnx', fp32: 'onnx/model.onnx',
  q4f16: 'onnx/model_q4f16.onnx', q2f16: 'onnx/model_q2f16.onnx', fp16: 'onnx/model_fp16.onnx',
};

// f16 activations come last: on Apple Metal (Chrome 154, transformers.js 4.3.1) Qwen2.5-0.5B's q4f16 graph decodes
// to word salad and fp16 loops, while q4 (f32 activations) answers correctly (test/browser/transformers.html,
// 2026-10-08). kohra hit the same fp16 overflow on WebGPU (kohra README, "Known constraints").
function pickDtype(model, caps) {
  const have = new Set(model.files.onnx.map((f) => f.name));
  const order = caps.f16 ? ['q4', 'q2', 'fp32', 'q4f16', 'q2f16', 'fp16'] : ['q4', 'q2', 'fp32'];
  return order.find((d) => have.has(FILES[d])) || null;
}

// Multimodal chat models: one ONNX graph per module (embed_tokens, decoder_model_merged, vision_encoder, audio_encoder)
// per precision, loaded through the model's conditional-generation class and its processor, as LocalMind does
// (LocalMind index.html loadMultimodal). Keyed by config.model_type. q4f16 first: LocalMind ships these at q4f16.
const MULTIMODAL = { gemma4: 'Gemma4ForConditionalGeneration', qwen3_5: 'Qwen3_5ForConditionalGeneration' };
const MM_SUFFIX = { fp32: '', fp16: '_fp16', q4: '_q4', q4f16: '_q4f16', q8: '_quantized' };
const mmModules = (model) => [...new Set(model.files.onnx.map((f) => /^onnx\/([a-z_]+?)(?:_fp16|_q4f16|_q4|_quantized)?\.onnx$/.exec(f.name)?.[1]).filter(Boolean))];
function pickMmDtype(model, caps) {
  const have = new Set(model.files.onnx.map((f) => f.name));
  const order = caps.f16 ? ['q4f16', 'q4', 'fp16', 'fp32'] : ['q4', 'fp32'];
  return order.find((d) => have.has(`onnx/decoder_model_merged${MM_SUFFIX[d]}.onnx`)) || null;
}
const mmBytes = (model, dtype) => mmModules(model).reduce((n, m) => n + graphBytes(model, `onnx/${m}${MM_SUFFIX[dtype]}.onnx`), 0);
const isMultimodal = (model) => model.architecture in MULTIMODAL && model.files.onnx.some((f) => f.name.startsWith('onnx/decoder_model_merged'));
// The chat tasks a multimodal repo is tagged with on the Hub; all run here as chat.
const MM_TASKS = ['text-generation', 'image-text-to-text', 'any-to-any'];

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
  // The library, pinned. Set transformersBackend.source to load another version.
  source: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/+esm',
  tasks: ['text-generation', ...Object.keys(PIPELINES)],

  canRun(model, caps, { task, budget = Infinity } = {}) {
    const fits = (bytes, ok) => (bytes > budget
      ? { ok: false, code: 'budget', need: bytes, reason: `the model stays on the GPU whole (${(bytes / 1e9).toFixed(2)} GB) and the budget is ${(budget / 1e9).toFixed(2)} GB` }
      : ok);
    if (!model.formats.includes('onnx')) return { ok: false, code: 'format', reason: 'no ONNX graph in the repo' };
    // kohra's diffusion graphs are ONNX too, but transformers.js generate() is autoregressive only.
    if (model.tags?.includes('text-diffusion')) return { ok: false, code: 'model', reason: 'a text-diffusion model; transformers.js generate() is autoregressive only' };
    // A repo with no Hub task tag (e.g. onnx-community/Ternary-Bonsai-4B-ONNX) is taken as text generation.
    if (isMultimodal(model)) {
      if (task && !MM_TASKS.includes(task)) return { ok: false, code: 'task', reason: `task ${task}; ${model.architecture} runs here as chat` };
      if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
      const dtype = pickMmDtype(model, caps);
      if (!dtype) return { ok: false, code: 'format', reason: 'no decoder_model_merged graph in a precision this back end loads' };
      const bytes = mmBytes(model, dtype);
      return fits(bytes, { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } });
    }
    const want = task || model.task || 'text-generation';
    if (!transformersBackend.tasks.includes(want)) return { ok: false, code: 'task', reason: `task ${want || 'unknown'}; this back end runs ${transformersBackend.tasks.join(', ')}` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (want !== 'text-generation') {
      const bytes = fp32Bytes(model);
      if (!bytes) return { ok: false, code: 'format', reason: 'no fp32 ONNX graph (onnx/model.onnx or encoder/decoder graphs)' };
      return fits(bytes, { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } });
    }
    const dtype = pickDtype(model, caps);
    if (!dtype) return { ok: false, code: 'format', reason: `no ONNX graph at ${Object.values(FILES).join(', ')}` };
    const bytes = graphBytes(model, FILES[dtype]);
    return fits(bytes, { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } });
  },

  // dtype overrides the precision pick (the repo must ship it). generate() takes repetitionPenalty (default 1, none).
  async load(model, { task, caps, dtype: forced, onProgress, signal }) {
    const want = task || model.task || 'text-generation';
    const lib = await import(transformersBackend.source);
    const progress_callback = onProgress && ((p) => onProgress({ backend: 'transformers', ...p }));
    if (isMultimodal(model)) return loadMultimodal(lib, model, forced || pickMmDtype(model, caps), { progress_callback, signal });
    if (want !== 'text-generation') return loadPipeline(lib, model, want, { progress_callback, signal });
    const dtype = forced || pickDtype(model, caps);
    const gen = await lib.pipeline('text-generation', model.repo, {
      device: 'webgpu', dtype, revision: model.revision || 'main', progress_callback,
    });
    signal?.throwIfAborted();
    return {
      task: 'text-generation',
      generate: (messages, { maxNewTokens = 256, repetitionPenalty = 1, signal } = {}) => stream(lib, gen.tokenizer, signal,
        (streamer, stopping_criteria) => gen(messages, { max_new_tokens: maxNewTokens, do_sample: false, repetition_penalty: repetitionPenalty, streamer, stopping_criteria })),
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

// transformers.js streams through a callback; this turns one generate call into the contract's async iterable of
// { text, tokens }, with the signal wired to an InterruptableStoppingCriteria.
async function* stream(lib, tokenizer, signal, run) {
  const queue = [];
  let wake = () => {}, done = false, failed = null, text = '', tokens = 0;
  const streamer = new lib.TextStreamer(tokenizer, {
    skip_prompt: true, skip_special_tokens: true,
    callback_function: (t) => { text += t; queue.push({ text, tokens }); wake(); },
    token_callback_function: () => { tokens++; },
  });
  const stop = new lib.InterruptableStoppingCriteria();
  signal?.addEventListener('abort', () => stop.interrupt(), { once: true });
  run(streamer, stop).catch((e) => { failed = e; }).finally(() => { done = true; wake(); });
  while (true) {
    if (queue.length) { yield queue.shift(); continue; }
    if (done) break;
    await new Promise((r) => { wake = r; });
  }
  if (failed) throw failed;
}

// A multimodal chat model. Message content is a string or an array of blocks: { type: 'text', text },
// { type: 'image', image } (ImageData, RawImage or a URL) and, for Gemma 4, { type: 'audio', audio } (16 kHz
// Float32Array). The template sees placeholders; the processor gets the media, in order.
async function loadMultimodal(lib, model, dtype, { progress_callback, signal }) {
  const opts = { revision: model.revision || 'main', progress_callback };
  const processor = await lib.AutoProcessor.from_pretrained(model.repo, opts);
  const net = await lib[MULTIMODAL[model.architecture]].from_pretrained(model.repo, { ...opts, dtype, device: 'webgpu' });
  signal?.throwIfAborted();
  // One-token warmup compiles the kernels before the first reply (LocalMind does the same).
  await net.generate({ ...(await processor('a', null, null, { add_special_tokens: false })), max_new_tokens: 1 });
  const toImage = async (x) => (typeof ImageData !== 'undefined' && x instanceof ImageData ? new lib.RawImage(x.data, x.width, x.height, 4)
    : typeof x === 'string' ? lib.load_image(x) : x);
  return {
    task: 'text-generation',
    generate: (messages, { maxNewTokens = 256, repetitionPenalty = 1, signal } = {}) => stream(lib, processor.tokenizer, signal, async (streamer, stopping_criteria) => {
      const images = [], audios = [];
      const plain = messages.map((m) => (typeof m.content === 'string' ? m : {
        ...m,
        content: m.content.map((b) => {
          if (b.type === 'image') { images.push(b.image); return { type: 'image' }; }
          if (b.type === 'audio') { audios.push(b.audio); return { type: 'audio' }; }
          return b;
        }),
      }));
      const prompt = processor.apply_chat_template(plain, { add_generation_prompt: true, enable_thinking: false });
      const imgs = await Promise.all(images.map(toImage));
      const one = (a) => (a.length ? (a.length === 1 ? a[0] : a) : null);
      const inputs = await processor(prompt, one(imgs), one(audios), { add_special_tokens: false });
      return net.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: false, repetition_penalty: repetitionPenalty, streamer, stopping_criteria });
    }),
    dispose: () => net.dispose?.(),
  };
}
