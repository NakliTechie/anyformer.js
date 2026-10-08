// B2, correctness: the same model, engine and prompt through LocalMind directly and through anyformer, greedy; the
// texts must match byte for byte. LocalMind is served from the same origin as anyformer's test pages, so both share
// the browser's model stores (the same engines use the same cache names) and nothing downloads twice.
//   node bench/parity.mjs --localmind ~/Code/naklios-universe/LocalMind [--rows key,key] [--n 64] [--out file]
// Each row runs in its own profile and port (see ROWS): the localmind profile holds most models, the chat profile
// holds Gemma 4 26B-A4B. Exits 0 when every row matches, 1 otherwise.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { realpathSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, extname, normalize, sep } from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const ROOT = realpathSync(new URL('..', import.meta.url).pathname);
const LM = realpathSync((arg('localmind', join(homedir(), 'Code/naklios-universe/LocalMind'))).replace(/^~/, homedir()));
const N = Number(arg('n', 64));
const PROMPT = 'Explain how a hash map works: hashing, buckets, collisions and resizing.';   // LocalMind's bench prose prompt
const LMP = { port: 8193, profile: join(homedir(), '.cache/anyformer-localmind-profile') };
const CHATP = { port: 8192, profile: join(homedir(), '.cache/anyformer-chat-profile') };
// LocalMind MODELS key → the Hub repo anyformer loads, its budget, and the profile holding the weights. dtype and rp
// repeat what LocalMind's MODELS entry sets (dtype, genConfig.repetition_penalty), which its API keeps at temperature 0.
export const ROWS = [
  { key: 'lfm2-230m-webgpu', repo: 'LiquidAI/LFM2.5-230M-GGUF', budget: 2.5, ...LMP },
  { key: 'gemma4-e2b-webgpu', repo: 'google/gemma-4-E2B-it-qat-mobile-transformers', budget: 2.5, ...LMP },
  { key: 'gemma4-e4b-webgpu', repo: 'google/gemma-4-E4B-it-qat-mobile-transformers', budget: 4, ...LMP },
  { key: 'ternary-bonsai-2-27b-webgpu', repo: 'prism-ml/Ternary-Bonsai-2-27B-gguf', budget: 7, ...LMP },
  { key: 'qwen35-4b', repo: 'onnx-community/Qwen3.5-4B-ONNX-OPT', budget: 8, dtype: 'q4f16', rp: 1.05, ...LMP },
  { key: 'gemma4-e2b', repo: 'onnx-community/gemma-4-E2B-it-ONNX', budget: 8, dtype: 'q4f16', rp: 1.1, ...LMP },
  { key: 'gemma4-e4b', repo: 'onnx-community/gemma-4-E4B-it-ONNX', budget: 8, dtype: 'q4f16', rp: 1.1, ...LMP },
  { key: 'lfm2-1.2b', repo: 'LiquidAI/LFM2.5-1.2B-Instruct-ONNX', budget: 2.5, dtype: 'q4f16', rp: 1.05, ...LMP },
  { key: 'bonsai-ternary-4b', repo: 'onnx-community/Ternary-Bonsai-4B-ONNX', budget: 2.5, dtype: 'q2f16', rp: 1.05, ...LMP },
  // LocalMind loads this engine without a budget (a 4 GiB expert pool) at contextSize 4096; 7.4 GB at 4096 tokens gives
  // anyformer the same pool (about 3.04 GB fixed + 4.29 GB of experts).
  { key: 'gemma4-26b-a4b-ssd', repo: 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf', budget: 7.4, ctx: 4096, ...CHATP },
];
const pick = arg('rows') ? arg('rows').split(',') : null;
const rows = ROWS.filter((r) => !pick || pick.includes(r.key));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.png': 'image/png' };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.error(new Date().toTimeString().slice(0, 8), ...a);

// /localmind/… serves LocalMind; everything else serves anyformer.
export function serve(port) {
  const server = createServer(async (req, res) => {
    try {
      let path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
      let base = ROOT;
      if (path === '/localmind' || path.startsWith('/localmind/')) { base = LM; path = path.slice('/localmind'.length) || '/'; }
      if (path.endsWith('/')) path += 'index.html';
      const p = join(base, path);
      if (!p.startsWith(base + sep) || !(await stat(p)).isFile()) { res.writeHead(404).end(); return; }
      res.writeHead(200, { 'Content-Type': types[extname(p)] || 'application/octet-stream' }).end(await readFile(p));
    } catch { if (!res.headersSent) res.writeHead(404); res.end(); }
  }).listen(port, '127.0.0.1');
  return new Promise((r) => server.once('listening', () => r(server)));
}

export async function browser(port, profile) {
  const cdpPort = 9600 + (port % 300);
  const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    ['--headless=new', '--no-first-run', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', `--remote-debugging-port=${cdpPort}`, 'about:blank'], { stdio: 'ignore' });
  let ws;
  for (let k = 0; k < 150 && !ws; k++) { try { const t = (await (await fetch(`http://127.0.0.1:${cdpPort}/json`)).json()).find((x) => x.type === 'page'); ws = new WebSocket(t.webSocketDebuggerUrl); } catch { await sleep(200); } }
  await new Promise((r) => { ws.onopen = r; });
  let id = 0; const pend = new Map();
  ws.onmessage = (e) => { const m = JSON.parse(e.data); if (pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
  const send = (method, params = {}) => new Promise((r) => { const n = ++id; pend.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
  const ev = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
    return r.result?.result?.value;
  };
  await send('Page.enable');
  return { send, ev, close: () => { try { chrome.kill(); } catch {} } };
}

// LocalMind direct, through its page API, as scripts/bench-engines.mjs drives it (API on, DFlash off, PLE resident).
export async function localmindReply(b, port, key, prompt, n, runs = 1) {
  const seed = `(() => { try { localStorage.setItem('lm_api_enabled', '1'); localStorage.setItem('lm_dflash', '0'); localStorage.setItem('lm_gemma4_ple', '0'); } catch (e) {} })()`;
  const { identifier } = (await b.send('Page.addScriptToEvaluateOnNewDocument', { source: seed })).result;
  await b.send('Page.navigate', { url: `http://127.0.0.1:${port}/localmind/index.html?bench=${Date.now()}` });
  for (let i = 0; i < 240 && !(await b.ev('!!(window.localmind && window.localmind.ready)').catch(() => false)); i++) await sleep(1000);
  await b.ev(`window.localmind.load(${JSON.stringify(key)})`);
  const out = [];
  for (let r = 0; r < runs; r++) out.push(await b.ev(`(async () => {
    const t0 = performance.now(); let first = 0, last = 0, text = '';
    const stream = await window.localmind.chat.completions.create({ messages: [{ role: 'user', content: ${JSON.stringify(prompt)} }], stream: true, temperature: 0, max_tokens: ${n} });
    for await (const ch of stream) { const d = ch.choices?.[0]?.delta?.content; if (d) { const now = performance.now(); if (!first) first = now; last = now; text += d; } }
    return { ttftMs: first - t0, decodeMs: last - first, text };
  })()`));
  await b.send('Page.removeScriptToEvaluateOnNewDocument', { identifier });
  return out;
}

// anyformer, through test/browser/answer.html on the same origin.
export async function anyformerReply(b, port, row, prompt, n, runs = 1, minutes = 60) {
  const q = new URLSearchParams({ repo: row.repo, budget: row.budget, n, prompt, runs, ...(row.dtype && { dtype: row.dtype }), ...(row.rp && { rp: row.rp }), ...(row.ctx && { ctx: row.ctx }), ...(arg('tjs') && { tjs: arg('tjs') }) });
  await b.send('Page.navigate', { url: `http://127.0.0.1:${port}/test/browser/answer.html?${q}` });
  for (const t0 = Date.now(); Date.now() - t0 < minutes * 60e3; await sleep(1000)) {
    const v = await b.ev('window.result || null').catch(() => null);
    if (v) { if (!v.ok) throw new Error(v.error); return v.runs; }
  }
  throw new Error('anyformer page timed out');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const results = [];
  for (const groupPort of [...new Set(rows.map((r) => r.port))]) {
    const group = rows.filter((r) => r.port === groupPort);
    const server = await serve(groupPort);
    const b = await browser(groupPort, group[0].profile);
    for (const r of group) {
      try {
        log(`${r.key}: LocalMind`);
        const [lm] = await localmindReply(b, r.port, r.key, PROMPT, N);
        log(`${r.key}: anyformer`);
        const [af] = await anyformerReply(b, r.port, r, PROMPT, N);
        let at = 0; while (at < lm.text.length && lm.text[at] === af.text[at]) at++;
        const identical = lm.text === af.text && lm.text.trim().length >= 20;   // two empty replies are not a match
        results.push({ key: r.key, repo: r.repo, identical, sameUpTo: at, lmChars: lm.text.length, afChars: af.text.length, ...(identical ? {} : { localmind: lm.text, anyformer: af.text }) });
        log(`${r.key}: ${identical ? 'identical' : `differs at char ${at}`}`);
      } catch (e) {
        results.push({ key: r.key, repo: r.repo, error: String(e.message || e).slice(0, 300) });
        log(`${r.key}: ERROR ${e.message}`);
      }
    }
    b.close(); server.close();
    await sleep(1500);
  }
  console.table(results.map((r) => ({ key: r.key, result: r.error ? `error: ${r.error.slice(0, 60)}` : r.identical ? 'identical' : `differs at ${r.sameUpTo}` })));
  if (arg('out')) writeFileSync(arg('out'), JSON.stringify({ date: new Date().toISOString(), prompt: PROMPT, n: N, results }, null, 1));
  process.exit(results.every((r) => r.identical) ? 0 : 1);
}
