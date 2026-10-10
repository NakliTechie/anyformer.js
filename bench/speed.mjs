// B3, speed, by LocalMind's method (LocalMind scripts/bench-engines.mjs): the prose prompt, 256 new tokens, greedy,
// 3 runs with run 1 as warm-up, the median of the rest; output tokens counted exactly, in the page, with the model's
// own tokenizer.json (special tokens excluded). Each row measures LocalMind direct and anyformer back to back on the
// same origin, profile and engine, so the gap is anyformer's overhead. Rows alternate which side goes first: the second
// side meets a warmer machine (2026-10-10: Bonsai 2 27B's second side sagged 25.1 → 20.7 tok/s). Close other GPU work first.
//   node bench/speed.mjs [--rows key,key] [--runs 3] [--n 256] [--out file] [--flip]
import { writeFileSync } from 'node:fs';
import { ROWS, serve, browser, localmindReply, anyformerReply } from './parity.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const RUNS = Number(arg('runs', 3)), N = Number(arg('n', 256));
const PROMPT = 'Explain how a hash map works: hashing, buckets, collisions and resizing.';
// tokenizer.json per model family, as LocalMind's bench uses them.
const TOK = {
  'lfm2-230m-webgpu': 'LiquidAI/LFM2.5-1.2B-Instruct-ONNX', 'lfm2-1.2b': 'LiquidAI/LFM2.5-1.2B-Instruct-ONNX',
  'gemma4-e2b-webgpu': 'onnx-community/gemma-4-E2B-it-ONNX', 'gemma4-e4b-webgpu': 'onnx-community/gemma-4-E2B-it-ONNX',
  'gemma4-e2b': 'onnx-community/gemma-4-E2B-it-ONNX', 'gemma4-e4b': 'onnx-community/gemma-4-E2B-it-ONNX',
  'gemma4-26b-a4b-ssd': 'onnx-community/gemma-4-E2B-it-ONNX',
  'qwen35-4b': 'onnx-community/Qwen3.5-4B-ONNX-OPT', 'ternary-bonsai-2-27b-webgpu': 'Qwen/Qwen3.8-27B',
  'bonsai-ternary-4b': 'onnx-community/Ternary-Bonsai-4B-ONNX',
};
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const log = (...a) => console.error(new Date().toTimeString().slice(0, 8), ...a);
const pick = arg('rows') ? arg('rows').split(',') : null;
const rows = ROWS.filter((r) => !pick || pick.includes(r.key));

async function count(b, key, texts) {
  return b.ev(`(async () => {
    const { Tokenizer } = await import('https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/+esm');
    const get = async (p) => { for (const f of [p, 'onnx/' + p]) { const r = await fetch('https://huggingface.co/${TOK[key]}/resolve/main/' + f); if (r.ok) return r.json(); } return {}; };
    const tj = await get('tokenizer.json'), tc = await get('tokenizer_config.json');
    const t = new Tokenizer(tj, tc), special = new Set((tj.added_tokens || []).filter((a) => a.special).map((a) => a.content));
    return ${JSON.stringify(texts)}.map((s) => t.encode(s, { addSpecialTokens: false }).tokens.filter((x) => !special.has(x)).length);
  })()`);
}

function summarize(runs, counts) {
  const per = runs.map((x, i) => ({ tokens: counts[i], ttftMs: Math.round(x.ttftMs), tokPerSec: +((counts[i] - 1) / (x.decodeMs / 1000)).toFixed(1) }));
  const warm = per.length > 1 ? per.slice(1) : per;
  return { tokPerSec: +median(warm.map((x) => x.tokPerSec)).toFixed(1), ttftMs: Math.round(median(warm.map((x) => x.ttftMs))), runs: per, identicalRuns: new Set(runs.map((x) => x.text)).size === 1 };
}

const results = [];
for (const port of [...new Set(rows.map((r) => r.port))]) {
  const group = rows.filter((r) => r.port === port);
  const server = await serve(port);
  const b = await browser(port, group[0].profile);
  for (const r of group) {
    try {
      const sides = {
        localmind: async () => { const x = await localmindReply(b, port, r.key, PROMPT, N, RUNS); return summarize(x, await count(b, r.key, x.map((y) => y.text))); },
        anyformer: async () => { const x = await anyformerReply(b, port, r, PROMPT, N, RUNS); return summarize(x, await count(b, r.key, x.map((y) => y.text))); },
      };
      // --flip starts the alternation with anyformer, so two passes put every row's sides in both orders.
      const flip = process.argv.includes('--flip') ? 1 : 0;
      const order = (results.length + flip) % 2 ? ['anyformer', 'localmind'] : ['localmind', 'anyformer'];
      const got = {};
      for (const side of order) { got[side] = await sides[side](); log(`${r.key}: ${side} ${got[side].tokPerSec} tok/s`); }
      results.push({ key: r.key, repo: r.repo, first: order[0], localmind: got.localmind, anyformer: got.anyformer, ratio: +(got.anyformer.tokPerSec / got.localmind.tokPerSec).toFixed(3) });
    } catch (e) {
      results.push({ key: r.key, repo: r.repo, error: String(e.message || e).slice(0, 300) });
      log(`${r.key}: ERROR ${e.message}`);
    }
  }
  b.close(); server.close();
  await new Promise((res) => setTimeout(res, 1500));
}
console.table(results.map((r) => ({ key: r.key, 'LocalMind tok/s': r.localmind?.tokPerSec ?? r.error, 'anyformer tok/s': r.anyformer?.tokPerSec, 'anyformer / LocalMind': r.ratio, 'TTFT ms (LM / af)': r.localmind ? `${r.localmind.ttftMs} / ${r.anyformer.ttftMs}` : '' })));
if (arg('out')) writeFileSync(arg('out'), JSON.stringify({ date: new Date().toISOString(), prompt: PROMPT, n: N, runs: RUNS, results }, null, 1));
