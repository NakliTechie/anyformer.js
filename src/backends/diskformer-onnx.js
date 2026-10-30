// diskformer-onnx: ONNX exports that transformers.js has no model class for, run by a diskformer hand engine on
// onnxruntime-web's WebGPU EP. The engine loads from the CDN at load() time, at the commit src/sources.js pins (it
// pins its own onnxruntime-web and transformers.js tokenizer versions); set diskformerOnnxBackend.source to load
// another copy. Model files go to Cache Storage 'diskformer-models' on the first load and come from there later.
//
// When transformers.js adds a model class for one of these architectures, its entry moves to the transformers back
// end (which ranks first in BACKENDS) and leaves this table.

import { graphBytes } from './transformers.js';
import { SOURCES } from '../sources.js';

// Keyed by config.model_type. graphs: the ONNX files the engine reads (each with its _data file); json: the small files.
export const ONNX_ENGINES = {
  // LiquidAI LFM3-Omni-2B: text, image and speech in (16 kHz Float32Array, at most 30 s a clip), text out. transformers.js
  // 4.3.2 has no lfm3_omni class; swap to transformers.js when it adds the model. diskformer's gate
  // (test/browser/lfm3-omni.html, onnxruntime-web 1.30.0): prompt ids and 32 greedy tokens equal Liquid's runner on
  // onnxruntime's CPU EP for text, image and speech, 3/3.
  lfm3_omni: {
    engine: 'lfm3_omni.js', Engine: 'Lfm3Omni',
    graphs: ['onnx/decoder_q4f16.onnx', 'onnx/vision_encoder_q8.onnx', 'onnx/audio_encoder.onnx'],
    json: ['config.json', 'generation_config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'],
  },
};

const missing = (model, e) => {
  const have = new Set(model.files.all.map((f) => f.name));
  return [...e.graphs.flatMap((g) => [g, `${g}_data`]), ...e.json].filter((f) => !have.has(f));
};
const bytesOf = (model, e) => e.graphs.reduce((a, g) => a + graphBytes(model, g), 0) +
  model.files.all.filter((f) => e.json.includes(f.name)).reduce((a, f) => a + (f.size || 0), 0);

// An image URL becomes an ImageBitmap (the engine takes ImageData, { data, width, height } or anything drawImage takes).
const toImage = async (x) => (typeof x === 'string' ? createImageBitmap(await (await fetch(x)).blob()) : x);
const media = (messages) => Promise.all(messages.map(async (m) => (typeof m.content === 'string' ? m : {
  ...m, content: await Promise.all(m.content.map(async (b) => (b.type === 'image' ? { ...b, image: await toImage(b.image) } : b))),
})));

export const diskformerOnnxBackend = {
  id: 'diskformer-onnx',
  // LFM3-Omni's repo is tagged any-to-any; the engine writes text only.
  tasks: ['any-to-any', 'text-generation', 'image-text-to-text'],
  source: SOURCES.diskformer,
  // The folder the engine reads the repo layout from; null is Hugging Face at the model's revision. A test points it at
  // a local server.
  url: null,

  canRun(model, caps, { task } = {}) {
    const e = ONNX_ENGINES[model.architecture];
    if (!e) return { ok: false, code: 'model', reason: `no diskformer ONNX engine for ${model.architecture || 'this architecture'}; engines cover ${Object.keys(ONNX_ENGINES).join(', ')}` };
    if (!model.formats.includes('onnx')) return { ok: false, code: 'format', reason: 'no ONNX graphs in the repo' };
    if (task && !diskformerOnnxBackend.tasks.includes(task)) return { ok: false, code: 'task', reason: `task ${task}; this engine writes text (${diskformerOnnxBackend.tasks.join(', ')})` };
    const gone = missing(model, e);
    if (gone.length) return { ok: false, code: 'format', reason: `the engine reads ${gone.join(', ')}, which the repo lacks` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (!caps.f16) return { ok: false, code: 'f16', reason: 'the GPU lacks shader-f16 (the q4f16 decoder runs fp16 activations)' };
    const bytes = bytesOf(model, e);
    return { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } };
  },

  async load(model, { onProgress, signal }) {
    const e = ONNX_ENGINES[model.architecture];
    const mod = await import(`${diskformerOnnxBackend.source}/${e.engine}`);
    const baseUrl = diskformerOnnxBackend.url ? diskformerOnnxBackend.url(model) : `https://huggingface.co/${model.repo}/resolve/${model.revision || 'main'}/`;
    const engine = await mod[e.Engine].load({ baseUrl, onProgress: onProgress && ((p) => onProgress({ backend: 'diskformer-onnx', ...p })) });
    if (signal?.aborted) { await engine.dispose(); signal.throwIfAborted(); }
    return {
      task: 'text-generation',
      engine,
      // Where the files came from: the engine's Cache Storage entries are keyed by these URLs
      // (s.engine.constructor.cached({ baseUrl }) / deleteCache({ baseUrl })).
      baseUrl,
      // Content is a string or blocks: { type: 'text', text }, { type: 'image', image } (ImageData, ImageBitmap, a URL,
      // or { data, width, height }) and { type: 'audio', audio } (16 kHz mono Float32Array). thinking: false (the
      // default, as on every back end here) answers directly. tools: the chat template's tool list; tool calls stay
      // raw text in the reply (LFM3 writes Qwen3-Coder XML), for the caller to parse. An abort ends the reply quietly.
      async *generate(messages, { maxNewTokens = 512, signal, thinking = false, tools, repetitionPenalty } = {}) {
        const queue = [];
        let wake = () => {}, done = false, failed = null, text = '', tokens = 0;
        const msgs = await media(messages);
        engine.generate(msgs, {
          max_new_tokens: maxNewTokens, do_sample: false, enable_thinking: thinking === true, signal,
          ...(tools && { tools }), ...(repetitionPenalty && { repetition_penalty: repetitionPenalty }),
          onToken: () => { tokens++; },
          onText: (delta) => { text += delta; queue.push({ text, tokens }); wake(); },
        }).catch((err) => { failed = err; }).finally(() => { done = true; wake(); });
        while (true) {
          if (queue.length) { yield queue.shift(); continue; }
          if (done) break;
          await new Promise((r) => { wake = r; });
        }
        if (failed) throw failed;
      },
      dispose: () => engine.dispose(),
    };
  },
};
