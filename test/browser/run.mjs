// Runs one test page in headless Chrome and prints window.result; exit 0 pass, 1 fail, 2 timeout.
//   node test/browser/run.mjs <page> [--port N] [--profile dir] [--minutes N] [--budget GB] [--query a=b&c=d] [--models dir,…]
// <page> is a test page name (test/browser/<page>.html) or a path from the repo root (examples/chat/index.html).
// kohra and diskformer load at the commits src/sources.js pins; --main loads each repo's current main commit
// instead (looked up on GitHub, so jsDelivr's cached @main never answers), to test a pin move before making it.
// diskformer-chat keeps a fixed port and profile by default: OPFS belongs to the origin, so the 14.4 GB model downloads
// once across runs.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, stat, mkdtemp } from 'node:fs/promises';
import { rmSync, createReadStream } from 'node:fs';
import { realpathSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join, extname, normalize, sep } from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const page = process.argv[2] || '';
const path = page.includes('/') ? page.replace(/^\.?\//, '') : `test/browser/${page.replace(/\.html$/, '')}.html`;
const name = path.replace(/^.*\//, '').replace(/\.html$/, '');
if (!page) { console.error('usage: node test/browser/run.mjs <page>'); process.exit(2); }
const chat = name === 'diskformer-chat';
const root = realpathSync(new URL('../..', import.meta.url).pathname);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };

// --models <dir>[,<dir>…]: serve the files in those folders at /models/<path> (a repo layout keeps its subfolders), with
// Range requests, so a test can copy a model into OPFS from this disk instead of Hugging Face (llama-gate's ?local=1).
const modelDirs = (arg('models') || '').split(',').filter(Boolean);
async function serveModel(req, res, file) {
  let p = null;
  for (const d of modelDirs) { const c = join(d, file); if (c.startsWith(join(d, sep)) && await stat(c).then((s) => s.isFile(), () => false)) { p = c; break; } }
  if (!p) { res.writeHead(404).end(); return; }
  const size = (await stat(p)).size;
  const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '');
  const start = m ? +m[1] : 0, end = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  res.writeHead(m ? 206 : 200, { 'Content-Type': 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1,
    ...(m && { 'Content-Range': `bytes ${start}-${end}/${size}` }) });
  createReadStream(p, { start, end }).pipe(res);
}

const server = createServer(async (req, res) => {
  try {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (path.startsWith('/models/') && modelDirs.length) { await serveModel(req, res, path.slice('/models/'.length)); return; }
    const p = join(root, path);
    if (!p.startsWith(root + sep) || !(await stat(p)).isFile()) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'Content-Type': types[extname(p)] || 'application/octet-stream' }).end(await readFile(p));
  } catch { if (!res.headersSent) res.writeHead(404); res.end(); }
}).listen(Number(arg('port', chat ? 8192 : 0)), '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const port = server.address().port, cdp = 9500 + (port % 400);

const head = async (repo) => (await (await fetch(`https://api.github.com/repos/NakliTechie/${repo}/commits/main`)).json()).sha;
const main = process.argv.includes('--main');
const [kohra, diskformer] = main ? await Promise.all([head('kohra'), head('diskformer.js')]) : [];
const q = new URLSearchParams({ ...(main && { kohra, diskformer }), ...(arg('budget') && { budget: arg('budget') }), ...Object.fromEntries(new URLSearchParams(arg('query', ''))) });
if (main) console.error(`kohra main ${kohra.slice(0, 7)}, diskformer main ${diskformer.slice(0, 7)}`);

// A throwaway profile (no --profile, not diskformer-chat) is deleted on exit: its downloads are never reused.
const temporary = !arg('profile') && !chat;
const profile = arg('profile') || (chat ? join(homedir(), '.cache/anyformer-chat-profile') : await mkdtemp(join(tmpdir(), 'anyformer-test-')));
const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--no-first-run', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', `--remote-debugging-port=${cdp}`, 'about:blank'], { stdio: 'ignore' });
const stop = (code) => {
  try { chrome.kill(); } catch {}
  server.close();
  // Chrome holds files in the profile until it exits; wait for that before deleting it.
  const done = () => { if (temporary) rmSync(profile, { recursive: true, force: true }); process.exit(code); };
  if (chrome.exitCode !== null) done(); else { chrome.once('exit', done); setTimeout(done, 5000); }
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws;
for (let k = 0; k < 100 && !ws; k++) { try { const t = (await (await fetch(`http://127.0.0.1:${cdp}/json`)).json()).find((x) => x.type === 'page'); ws = new WebSocket(t.webSocketDebuggerUrl); } catch { await sleep(200); } }
await new Promise((r) => { ws.onopen = r; });
let id = 0; const pend = new Map(); ws.onmessage = (e) => { const m = JSON.parse(e.data); if (pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const n = ++id; pend.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const ev = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true })).result?.result?.value;

await send('Page.navigate', { url: `http://127.0.0.1:${port}/${path}?${q}` });
let last = '';
for (const t0 = Date.now(); Date.now() - t0 < Number(arg('minutes', chat ? 90 : 15)) * 60e3; await sleep(1000)) {
  const v = await ev('window.result || null');
  if (v) { console.log(JSON.stringify(v, null, 1)); stop(v.ok ? 0 : 1); }
  const p = await ev('window.progressText || ""');
  if (p && p !== last) { console.error(`${new Date().toTimeString().slice(0, 8)} ${p}`); last = p; }
}
console.error('timed out'); stop(2);
