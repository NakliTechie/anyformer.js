// diskformer-llama: mixture-of-experts GGUFs through diskformer's llama.cpp path (diskformer/llama: llama.cpp with
// WebGPU in a worker, the experts paged in from OPFS on demand). One engine for every architecture diskformer has
// checked in Chrome, so a new MoE model of a covered architecture needs no anyformer code. Where diskformer also has a
// hand-written engine for the model (the diskformer back end), that one carries a measured speed and ranks first.
// llama.js loads from the CDN at load() time, at the commit src/sources.js pins; set diskformerLlamaBackend.source to
// load another copy.

import { AnyformerError } from '../errors.js';
import { SOURCES } from '../sources.js';

// Architectures that pass diskformer's two checks (docs/coverage.md at the pinned commit): paging is exact natively,
// and the Chrome replies are compared with llama.cpp. Other models of the same architecture are expected to run.
// Left out: hy_v3 and afmoe (run, below diskformer's bar), laguna (unknown tokens), dense architectures (no experts
// to page), and Mixtral-style `llama` MoE (runs, too slow to use).
export const LLAMA_ARCHITECTURES = [
  'gemma4', 'qwen3moe', 'qwen3next', 'qwen35moe', 'qwen3vlmoe', 'deepseek2', 'nemotron_h_moe', 'ernie4_5-moe',
  'gpt-oss', 'cohere2moe', 'lfm2moe', 'mellum', 'bailingmoe2', 'granitehybrid', 'bailingmoe3', 'flint', 'olmoe',
];

// Quant preference by the token in the file name; the first one the repo has wins. The files diskformer checked are
// Q4_K_M, Q4_0, Q4_1, MXFP4 and Q3_K_M.
const QUANTS = ['Q4_K_M', 'Q4_0', 'Q4_1', 'MXFP4', 'Q3_K_M', 'Q4_K_S', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0'];

// The GGUF to load: the language model (no mmproj vision tower), one file (no split sets), the preferred quant.
export function pickGguf(model) {
  const lms = model.files.gguf.filter((f) => !/mmproj/i.test(f.name) && !/-\d{5}-of-\d{5}\.gguf$/.test(f.name));
  for (const q of QUANTS) {
    const f = lms.find((x) => new RegExp(`(^|[-_.])${q}([-_.]|$)`, 'i').test(x.name.replace(/\.gguf$/i, '')));
    if (f) return f;
  }
  return null;
}

// OPFS keeps llama.cpp's GGUFs by file name (gguf/<name>), with no repo. anyformer records name → repo here, so
// stored() can say which repo a file belongs to.
export const LLAMA_INDEX = 'anyformer-llama.json';
async function readIndex() {
  try { return JSON.parse(await (await (await (await navigator.storage.getDirectory()).getFileHandle(LLAMA_INDEX)).getFile()).text()); } catch { return {}; }
}
async function writeIndex(index) {
  const h = await (await navigator.storage.getDirectory()).getFileHandle(LLAMA_INDEX, { create: true });
  const w = await h.createWritable();
  await w.write(JSON.stringify(index));
  await w.close();
}
export { readIndex as readLlamaIndex };

// Chat content: a string, or text blocks joined. llama.cpp's chat here takes text only.
function textOnly(messages) {
  return messages.map((m) => {
    if (typeof m.content === 'string') return m;
    const other = m.content.find((b) => b.type !== 'text');
    if (other) throw new AnyformerError('INVALID_ARGUMENT', `diskformer-llama runs text only; got a ${other.type} block`);
    return { ...m, content: m.content.map((b) => b.text).join('\n') };
  });
}

export const diskformerLlamaBackend = {
  id: 'diskformer-llama',
  tasks: ['text-generation', 'image-text-to-text'],
  source: SOURCES.diskformerLlama,
  // Where a GGUF downloads from; a test points it at a local server.
  url: (model, file) => `${model.endpoint || 'https://huggingface.co'}/${model.repo}/resolve/${model.revision || 'main'}/${file.name.split('/').map(encodeURIComponent).join('/')}`,

  canRun(model, caps, { budget, task }) {
    if (!model.formats.includes('gguf')) return { ok: false, code: 'format', reason: 'no GGUF in the repo' };
    if (task && !diskformerLlamaBackend.tasks.includes(task)) return { ok: false, code: 'task', reason: `task ${task}; diskformer's llama.cpp path runs text generation only` };
    if (!LLAMA_ARCHITECTURES.includes(model.architecture)) return { ok: false, code: 'model', reason: `architecture ${model.architecture || 'unknown'} is not one diskformer's llama.cpp path has checked (${LLAMA_ARCHITECTURES.join(', ')})` };
    const f = pickGguf(model);
    if (!f) return { ok: false, code: 'model', reason: `no single-file GGUF at ${QUANTS.join(', ')} in the repo` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (!caps.memory64 || !caps.jspi) return { ok: false, code: 'webgpu', reason: 'the llama.cpp build needs wasm memory64 and JSPI (Chrome or Edge 137+)' };
    if (!caps.opfs) return { ok: false, code: 'opfs', reason: 'no OPFS (navigator.storage.getDirectory)' };
    // The budget is not checked here: llama.cpp plans its expert slots from it at load time, and refuses there.
    return { ok: true, estimate: { gpuBytes: budget, diskBytes: f.size, tokPerSec: null } };
  },

  // maxCtx sizes the KV cache; 1024 is the context diskformer's gates run at.
  async load(model, { budget, maxCtx = 1024, onProgress, signal }) {
    const f = pickGguf(model);
    const { Llama } = await import(diskformerLlamaBackend.source);
    const llama = new Llama();
    const say = (e) => onProgress?.({ backend: 'diskformer-llama', file: f.name, ...e });
    const stop = () => llama.abort();
    signal?.addEventListener('abort', stop, { once: true });
    let info;
    try {
      const have = (await llama.list()).find((m) => m.name === f.name && !m.partial);
      if (!have || have.size !== f.size) {
        say({ status: 'download', loaded: 0, total: f.size });
        await llama.download({ url: diskformerLlamaBackend.url(model, f), name: f.name, size: f.size },
          (p) => say({ status: 'download', loaded: p.copied, total: p.total }));
        signal?.throwIfAborted();
      }
      await writeIndex({ ...(await readIndex()), [f.name]: model.repo });
      say({ status: 'load' });
      info = await llama.load({ name: f.name, budget, ctx: maxCtx }, (p) => say({ status: 'plan', plan: p }));
      signal?.throwIfAborted();
    } catch (e) {
      llama.terminate();
      throw e;
    } finally {
      signal?.removeEventListener('abort', stop);
    }
    say({ status: 'ready', slots: info.slots });
    return {
      task: 'text-generation',
      llama,
      info,
      async *generate(messages, { maxNewTokens = 512, signal } = {}) {
        const chunks = [];
        let wake = null, finished = false, failed = null, result = null, text = '', tokens = 0;
        const nudge = () => { wake?.(); wake = null; };
        const abort = () => llama.abort();
        signal?.addEventListener('abort', abort, { once: true });
        const job = llama.chat(textOnly(messages), {
          n: maxNewTokens,
          onPiece: (s) => { text += s; tokens++; chunks.push({ text, tokens }); nudge(); },
        }).then((r) => { result = r; }, (e) => { failed = e; }).finally(() => { finished = true; nudge(); });
        try {
          for (;;) {
            while (chunks.length) yield chunks.shift();
            if (finished) break;
            await new Promise((r) => { wake = r; });
          }
          signal?.throwIfAborted();
          if (failed) throw failed;
          const s = result.stats || {};
          const decode = s.decode_tokens && s.decode_ms ? s.decode_tokens / (s.decode_ms / 1000) : null;
          yield { text: result.text, tokens: s.decode_tokens ?? tokens, ...(decode && { tokPerSec: decode }) };
        } finally {
          signal?.removeEventListener('abort', abort);
          // The caller stopped reading early: stop the reply so the next call does not queue behind it.
          if (!finished) { llama.abort(); await job; }
        }
      },
      // A continuation of an already formatted prompt, greedy: what diskformer's gate replays.
      complete: (prompt, { maxNewTokens = 64 } = {}) => llama.complete(prompt, { n: maxNewTokens }),
      dispose: () => llama.terminate(),
    };
  },
};
