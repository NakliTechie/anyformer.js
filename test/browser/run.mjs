// Runs one test page in headless Chrome and prints window.result; exit 0 pass, 1 fail, 2 timeout.
//   node test/browser/run.mjs <page> [--port N] [--profile dir] [--minutes N] [--budget GB]
// kohra and diskformer load at their current main commit (looked up on GitHub), not jsDelivr's cached @main.
// diskformer-chat keeps a fixed port and profile by default: OPFS belongs to the origin, so the 14.4 GB model downloads
// once across runs.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, stat, mkdtemp } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join, extname, normalize, sep } from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const name = (process.argv[2] || '').replace(/^.*\//, '').replace(/\.html$/, '');
if (!name) { console.error('usage: node test/browser/run.mjs <page>'); process.exit(2); }
const chat = name === 'diskformer-chat';
const root = realpathSync(new URL('../..', import.meta.url).pathname);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };

const server = createServer(async (req, res) => {
  try {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    const p = join(root, path);
    if (!p.startsWith(root + sep) || !(await stat(p)).isFile()) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'Content-Type': types[extname(p)] || 'application/octet-stream' }).end(await readFile(p));
  } catch { if (!res.headersSent) res.writeHead(404); res.end(); }
}).listen(Number(arg('port', chat ? 8192 : 0)), '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const port = server.address().port, cdp = 9500 + (port % 400);

const head = async (repo) => (await (await fetch(`https://api.github.com/repos/NakliTechie/${repo}/commits/main`)).json()).sha;
const [kohra, diskformer] = await Promise.all([head('kohra'), head('diskformer.js')]);
const q = new URLSearchParams({ kohra, diskformer, ...(arg('budget') && { budget: arg('budget') }) });
console.error(`kohra main ${kohra.slice(0, 7)}, diskformer main ${diskformer.slice(0, 7)}`);

const profile = arg('profile') || (chat ? join(homedir(), '.cache/anyformer-chat-profile') : await mkdtemp(join(tmpdir(), 'anyformer-test-')));
const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--no-first-run', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', `--remote-debugging-port=${cdp}`, 'about:blank'], { stdio: 'ignore' });
const stop = (code) => { try { chrome.kill(); } catch {} server.close(); process.exit(code); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws;
for (let k = 0; k < 100 && !ws; k++) { try { const t = (await (await fetch(`http://127.0.0.1:${cdp}/json`)).json()).find((x) => x.type === 'page'); ws = new WebSocket(t.webSocketDebuggerUrl); } catch { await sleep(200); } }
await new Promise((r) => { ws.onopen = r; });
let id = 0; const pend = new Map(); ws.onmessage = (e) => { const m = JSON.parse(e.data); if (pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const n = ++id; pend.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const ev = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true })).result?.result?.value;

await send('Page.navigate', { url: `http://127.0.0.1:${port}/test/browser/${name}.html?${q}` });
let last = '';
for (const t0 = Date.now(); Date.now() - t0 < Number(arg('minutes', chat ? 90 : 15)) * 60e3; await sleep(1000)) {
  const v = await ev('window.result || null');
  if (v) { console.log(JSON.stringify(v, null, 1)); stop(v.ok ? 0 : 1); }
  const p = await ev('window.progressText || ""');
  if (p && p !== last) { console.error(`${new Date().toTimeString().slice(0, 8)} ${p}`); last = p; }
}
console.error('timed out'); stop(2);
