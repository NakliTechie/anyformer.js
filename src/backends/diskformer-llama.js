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

// The exact files diskformer checked (ggml/chat.html CATALOG and docs/coverage.md at the pinned commit). When a repo
// holds one, it is the default: the coverage numbers were measured on that file.
export const CHECKED_FILES = [
  'gemma-4-26B_q4_0-it.gguf', 'Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf', 'Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf',
  'Qwen3.6-35B-A3B-UD-Q3_K_M.gguf', 'Qwen3-VL-30B-A3B-Instruct-Q3_K_M.gguf', 'GLM-4.7-Flash-Q3_K_M.gguf',
  'NVIDIA-Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf', 'ERNIE-4.5-21B-A3B-Thinking-Q4_K_M.gguf', 'gpt-oss-20b-MXFP4.gguf',
  'North-Mini-Code-1.0-UD-Q3_K_M.gguf', 'LFM2.5-24B-A2B-QAD-Q4_0.gguf', 'LFM2-8B-A1B-Q4_K_M.gguf',
  'Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf', 'Ling-mini-2.0-Q4_K_M.gguf', 'granite-4.0-h-tiny-Q4_K_M.gguf',
  'Ling-3.0-tiny-Q4_K_M.gguf', 'flint-int4-q4_1.gguf', 'OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf',
  'Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf',
];

const SPLIT = /-(\d{5})-of-(\d{5})\.gguf$/i;

// Drafters diskformer has checked for a target GGUF (ggml/chat.html CATALOG at the pinned commit), keyed by the
// target's file name. With speculative: true the drafter downloads beside the target and llama.cpp speculates with
// it; replies stay greedy-identical (Nemotron 3.5 Bolt E4B + DFlash: 9/9 against plain greedy and llama-server on
// WebGPU). Off by default, because on WebGPU it is slower: Bolt 24.7 tok/s with the drafter, 30.0 without (Chrome,
// diskformer), since a 4-token verify costs about 3 single decodes.
export const DRAFTERS = {
  'Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf': { repo: 'nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash', name: 'Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf', size: 736812096 },
};

// The GGUF to load, as { name, size, parts: [{ name, size }] }: the language model (no mmproj vision tower), one file or
// every part of a split set (name-00001-of-0000n.gguf, named by its first part; size is the sum). The caller's file
// when given; else the file diskformer checked; else the preferred quant, a single file before a split set.
export function pickGguf(model, file = null) {
  const lms = model.files.gguf.filter((f) => !/mmproj/i.test(f.name));
  const sets = new Map();
  for (const f of lms.filter((x) => SPLIT.test(x.name))) {
    const base = f.name.replace(SPLIT, '');
    sets.set(base, [...(sets.get(base) || []), f]);
  }
  const candidates = [
    ...lms.filter((f) => !SPLIT.test(f.name)).map((f) => ({ name: f.name, size: f.size, parts: [f] })),
    // A set missing a part cannot load.
    ...[...sets.values()].filter((p) => p.length === Number(SPLIT.exec(p[0].name)[2])).map((p) => {
      const parts = [...p].sort((a, b) => a.name.localeCompare(b.name));
      return { name: parts[0].name, size: parts.reduce((n, f) => n + f.size, 0), parts };
    }),
  ];
  if (file) return candidates.find((f) => f.name === file) || null;
  const checked = candidates.find((f) => CHECKED_FILES.includes(opfsName(f)));
  if (checked) return checked;
  for (const q of QUANTS) {
    const f = candidates.find((x) => new RegExp(`(^|[-_.])${q}([-_.]|$)`, 'i').test(x.name.replace(SPLIT, '').replace(/\.gguf$/i, '')));
    if (f) return f;
  }
  return null;
}

// llama.cpp's GGUFs sit in OPFS as gguf/<file name>, without the repo's folders: the worker finds a split set's other
// parts by name beside the first.
const opfsName = (f) => f.name.split('/').pop();

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

  canRun(model, caps, { budget, task, file = null }) {
    if (!model.formats.includes('gguf')) return { ok: false, code: 'format', reason: 'no GGUF in the repo' };
    if (task && !diskformerLlamaBackend.tasks.includes(task)) return { ok: false, code: 'task', reason: `task ${task}; diskformer's llama.cpp path runs text generation only` };
    if (!LLAMA_ARCHITECTURES.includes(model.architecture)) return { ok: false, code: 'model', reason: `architecture ${model.architecture || 'unknown'} is not one diskformer's llama.cpp path has checked (${LLAMA_ARCHITECTURES.join(', ')})` };
    const f = pickGguf(model, file);
    if (!f) return { ok: false, code: 'model', reason: file ? `${file} is not a language-model GGUF in the repo (a split set is named by its first part, with every part present)` : `no GGUF at ${QUANTS.join(', ')} in the repo (a split set needs every part)` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (!caps.memory64 || !caps.jspi) return { ok: false, code: 'webgpu', reason: 'the llama.cpp build needs wasm memory64 and JSPI (Chrome or Edge 137+)' };
    if (!caps.opfs) return { ok: false, code: 'opfs', reason: 'no OPFS (navigator.storage.getDirectory)' };
    // The budget is not checked here: llama.cpp plans its expert slots from it at load time, and refuses there.
    return { ok: true, estimate: { gpuBytes: budget, diskBytes: f.size, tokPerSec: null } };
  },

  // maxCtx sizes the KV cache; 1024 is the context diskformer's gates run at.
  // speculative: true also loads the model's checked drafter (DRAFTERS); otherwise the target loads alone.
  async load(model, { budget, file = null, maxCtx = 1024, speculative, onProgress, signal }) {
    const f = pickGguf(model, file);
    const drafter = speculative === true ? DRAFTERS[opfsName(f)] ?? null : null;
    // The drafter is one more file to fetch; it lives in its own repo.
    const files = [...f.parts.map((p) => ({ name: opfsName(p), size: p.size, url: diskformerLlamaBackend.url(model, p) })),
      ...(drafter ? [{ name: drafter.name, size: drafter.size, url: diskformerLlamaBackend.url({ ...model, repo: drafter.repo, revision: null }, drafter) }] : [])];
    const total = files.reduce((n, x) => n + x.size, 0);
    const { Llama } = await import(diskformerLlamaBackend.source);
    const llama = new Llama();
    const say = (e) => onProgress?.({ backend: 'diskformer-llama', file: f.name, ...e });
    const stop = () => llama.abort();
    signal?.addEventListener('abort', stop, { once: true });
    let info;
    try {
      // Each part (and the drafter) downloads under its own name; progress counts bytes over all of them.
      let before = 0;
      for (const x of files) {
        const have = (await llama.list()).find((m) => m.name === x.name && !m.partial);
        if (!have || have.size !== x.size) {
          say({ status: 'download', loaded: before, total });
          await llama.download(x, (p) => say({ status: 'download', loaded: before + p.copied, total }));
          signal?.throwIfAborted();
        }
        before += x.size;
      }
      // The drafter is filed under the target's repo, so remove(repo) deletes both.
      await writeIndex({ ...(await readIndex()), ...Object.fromEntries(files.map((x) => [x.name, model.repo])) });
      say({ status: 'load' });
      info = await llama.load({ name: opfsName(f), budget, ctx: maxCtx, draft: drafter?.name ?? null }, (p) => say({ status: 'plan', plan: p }));
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
      // The drafter's file name when llama.cpp speculates with one.
      drafter: drafter?.name ?? null,
      // llama.cpp hands back text pieces, not token ids, so there are no think ids to report.
      think: null,
      // thinking: false (the default, as the other engines here) renders the template with thinking off, so a model that
      // reasons first answers directly; true lets it think.
      async *generate(messages, { maxNewTokens = 512, signal, thinking = false } = {}) {
        const chunks = [];
        let wake = null, finished = false, failed = null, result = null, text = '', tokens = 0;
        const nudge = () => { wake?.(); wake = null; };
        const abort = () => llama.abort();
        signal?.addEventListener('abort', abort, { once: true });
        const job = llama.chat(textOnly(messages), {
          n: maxNewTokens, thinking,
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
