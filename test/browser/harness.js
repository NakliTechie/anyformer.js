import { kohraBackend, diskformerBackend, localmindBackend, transformersBackend } from '../../index.js';

// ?kohra=<sha>&diskformer=<sha>&localmind=<sha> (set by run.mjs): load those commits instead of jsDelivr's cached @main.
const q = new URLSearchParams(location.search);
if (q.get('kohra')) kohraBackend.source = `https://cdn.jsdelivr.net/gh/NakliTechie/kohra@${q.get('kohra')}/kohra.js`;
// ?tjs=<version> loads that transformers.js release instead of the pinned one.
if (q.get('tjs')) transformersBackend.source = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${q.get('tjs')}/+esm`;
if (q.get('localmind')) localmindBackend.source = `https://cdn.jsdelivr.net/gh/NakliTechie/LocalMind@${q.get('localmind')}/src`;
if (q.get('diskformer')) diskformerBackend.source = `https://cdn.jsdelivr.net/gh/NakliTechie/diskformer.js@${q.get('diskformer')}/engines`;

// Shared by the test pages: progress to window.progressText, the outcome to window.result and the title.
export const out = document.body.appendChild(document.createElement('pre'));
export const log = (s) => { window.progressText = s; out.textContent += `${s}\n`; };
export async function run(name, fn) {
  try {
    const r = await fn();
    window.result = r;
  } catch (e) {
    window.result = { ok: false, error: `${e.name}: ${e.message}`, stack: e.stack };
  }
  log(JSON.stringify(window.result, null, 1));
  document.title = `${window.result.ok ? 'PASS' : 'FAIL'} ${name}`;
}
// One chat reply through anyformer, timed: the last chunk's text, token count, tok/s after the first chunk.
export async function reply(session, messages, maxNewTokens) {
  const t0 = performance.now();
  let text = '', tokens = 0, tFirst = 0, firstTokens = 0;
  for await (const c of session.generate(messages, { maxNewTokens })) {
    if (!tFirst) { tFirst = performance.now(); firstTokens = c.tokens; }
    text = c.text; tokens = c.tokens;
    if (c.tokPerSec) return { text, tokens, tokPerSec: +c.tokPerSec.toFixed(2), seconds: +((performance.now() - t0) / 1e3).toFixed(1) };
  }
  const secs = (performance.now() - tFirst) / 1e3;
  return { text, tokens, tokPerSec: tokens > firstTokens ? +((tokens - firstTokens) / secs).toFixed(2) : null, seconds: +((performance.now() - t0) / 1e3).toFixed(1) };
}
// An on-topic answer to the CPU question: it names what a CPU does, and no 6-word run repeats (a looping decode).
export function onTopic(t) {
  const words = t.toLowerCase().match(/[a-z']+/g) || [];
  const runs = words.slice(0, -5).map((_, i) => words.slice(i, i + 6).join(' '));
  return /instruction|process|calculat|comput|execut/i.test(t) && new Set(runs).size === runs.length;
}
