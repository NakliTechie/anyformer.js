// wllama: llama.cpp compiled to WebAssembly, for GGUF models no hand-written engine covers (any architecture its
// llama.cpp knows, e.g. openbmb/MiniCPM5-2B-GGUF). It offloads every layer to WebGPU when the browser has it and runs
// on the CPU otherwise, so it is anyformer's one path that needs no WebGPU. Listed last in BACKENDS: an engine written
// for the architecture (localmind, diskformer) wins when one exists.

import { SOURCES } from '../sources.js';
import { splitToOpfs } from '../gguf-split.js';

// wllama reads files of at most 2 GB (one ArrayBuffer each) into a 32-bit WebAssembly memory of 4 GB. A model over
// 2 GB loads as shards: the repo's own split set when it has one, else shards anyformer writes to OPFS
// (src/gguf-split.js). The whole model, shards included, must stay under TOTAL_LIMIT to leave room for the context.
const FILE_LIMIT = 2 ** 31;
const TOTAL_LIMIT = 3.6e9;
const QUANTS = ['Q4_K_M', 'Q4_0', 'Q4_K_S', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0'];
const SPLIT = /-(\d{5})-of-(\d{5})\.gguf$/i;

// What to load: { kind: 'file' | 'split' | 'local-split', name, files: [{ name, size }], bytes }, or null.
function pickFile(model) {
  const lms = model.files.gguf.filter((f) => !/mmproj/i.test(f.name));
  const sets = new Map();
  for (const f of lms.filter((x) => SPLIT.test(x.name))) {
    const base = f.name.replace(SPLIT, '');
    sets.set(base, [...(sets.get(base) || []), f]);
  }
  for (const q of QUANTS) {
    const quant = new RegExp(`[-_.]${q}$`, 'i');
    const single = lms.find((x) => !SPLIT.test(x.name) && quant.test(x.name.replace(/\.gguf$/i, '')));
    if (single) return { kind: single.size > FILE_LIMIT ? 'local-split' : 'file', name: single.name, files: [single], bytes: single.size };
    for (const [base, parts] of sets) {
      const total = Number(SPLIT.exec(parts[0].name)[2]);
      if (quant.test(base) && parts.length === total) {
        const files = [...parts].sort((a, b) => a.name.localeCompare(b.name));
        return { kind: 'split', name: files[0].name, files, bytes: files.reduce((n, f) => n + f.size, 0) };
      }
    }
  }
  return null;
}

// wllama 3.4.1's stream ends one result early: the last chunk stays queued in the wasm side and would open the next
// request. Read the queue dry before and after every reply, as LocalMind does (its WLLAMA-DRAIN, index.html).
async function drain(w) {
  const proxy = w?.proxy;
  if (!proxy?.wllamaAction) return [];
  const out = [];
  for (let i = 0; i < 64; i++) {
    const r = await proxy.wllamaAction('get_result', { _name: 'gres_req' });
    const j = r?.data_json;
    if (!j || j === 'null') { if (!r?.has_more) break; continue; }
    let arr; try { arr = JSON.parse(j); } catch { break; }
    out.push(...(Array.isArray(arr) ? arr : [arr]));
  }
  return out;
}
const delta = (chunk) => chunk?.choices?.[0]?.delta?.content || '';

export const wllamaBackend = {
  id: 'wllama',
  tasks: ['text-generation'],
  source: SOURCES.wllama,

  canRun(model, caps, { budget, task } = {}) {
    if (!model.formats.includes('gguf')) return { ok: false, code: 'format', reason: 'no GGUF in the repo' };
    if (task && task !== 'text-generation' && task !== 'image-text-to-text') return { ok: false, code: 'task', reason: `task ${task}; wllama runs text generation` };
    const f = pickFile(model);
    if (!f) return { ok: false, code: 'model', reason: `no GGUF at ${QUANTS.join(', ')}` };
    if (f.bytes > TOTAL_LIMIT) return { ok: false, code: 'model', reason: `${f.name} is ${(f.bytes / 1e9).toFixed(2)} GB; wllama's 32-bit WebAssembly memory holds about ${TOTAL_LIMIT / 1e9} GB of model` };
    if (f.kind === 'local-split' && !caps.opfs) return { ok: false, code: 'opfs', reason: `${f.name} is over 2 GB and is split into OPFS before loading` };
    // On WebGPU the whole model sits on the GPU, so it must fit the budget; without WebGPU it runs on the CPU.
    const gpu = caps.webgpu && f.bytes <= budget;
    return { ok: true, estimate: { gpuBytes: gpu ? f.bytes : 0, diskBytes: f.bytes, tokPerSec: null } };
  },

  async load(model, { caps, budget, maxCtx = 4096, onProgress, signal }) {
    const f = pickFile(model);
    const { Wllama } = await import(`${wllamaBackend.source}/index.js`);
    const w = new Wllama({ default: `${wllamaBackend.source}/wasm/wllama.wasm` }, { logger: { debug() {}, log() {}, warn() {}, error: (m) => console.error('[wllama]', m) } });
    // isSupportWebGPU() may answer directly or with a promise, depending on the wllama version.
    let gpu = false;
    if (caps.webgpu && f.bytes <= budget) { try { gpu = !!(await w.isSupportWebGPU()); } catch { gpu = false; } }
    const rev = model.revision || 'main';
    const url = (name) => `https://huggingface.co/${model.repo}/resolve/${rev}/${name}`;
    const cfg = {
      n_ctx: maxCtx, n_gpu_layers: gpu ? 999 : 0,
      progressCallback: onProgress && (({ loaded, total }) => onProgress({ backend: 'wllama', status: 'download', file: f.name, loaded, total })),
    };
    if (f.kind === 'local-split') {
      const shards = await splitToOpfs({
        url: url(f.name), size: f.bytes, repo: model.repo, file: f.name, signal,
        key: `${model.repo}--${rev}--${f.name}`.replace(/[^A-Za-z0-9._-]+/g, '-'),
        onProgress: onProgress && ((p) => onProgress({ backend: 'wllama', ...p })),
      });
      await w.loadModel(shards, cfg);
    } else {
      await w.loadModelFromUrl(f.kind === 'split' ? f.files.map((x) => url(x.name)) : url(f.name), cfg);
    }
    signal?.throwIfAborted();
    return {
      task: 'text-generation',
      device: gpu ? 'webgpu' : 'cpu',
      async *generate(messages, { maxNewTokens = 256, signal } = {}) {
        const msgs = messages.map((m) => ({ role: m.role, content: typeof m.content === 'string' ? m.content : m.content.filter((b) => b.type === 'text').map((b) => b.text).join('\n') }));
        await drain(w).catch(() => {});
        const ctl = new AbortController();
        signal?.addEventListener('abort', () => ctl.abort(), { once: true });
        let text = '', tokens = 0;
        try {
          // Thinking off, as on every other back end: MiniCPM5 otherwise spends the reply in reasoning_content.
          const stream = await w.createChatCompletion({
            messages: msgs, stream: true, max_tokens: maxNewTokens, temperature: 0, abortSignal: ctl.signal,
            chat_template_kwargs: { enable_thinking: false },
          });
          for await (const c of stream) { const d = delta(c); if (d) { text += d; tokens++; yield { text, tokens }; } }
          for (const c of await drain(w)) { const d = delta(c); if (d) { text += d; tokens++; yield { text, tokens }; } }
        } catch (e) {
          if (!ctl.signal.aborted) throw e;
        }
      },
      dispose: () => w.exit(),
    };
  },
};
