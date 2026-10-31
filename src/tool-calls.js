// Tool calls for every back end: generate(messages, { tools }) takes OpenAI-style tools ({ type: 'function', function:
// { name, description, parameters } }), writes them into the conversation in the model family's own format, and reads
// the calls back out of the reply as OpenAI `tool_calls`. Assistant turns that carry `tool_calls` and `tool` turns that
// carry results are written back in the same format, so a multi-step tool loop needs no model-specific code.
//
// Formats:
//   lfm2    LFM2 / LFM2.5 (architectures lfm2, lfm2moe): the system turn gets "List of tools: <|tool_list_start|>[…]
//           <|tool_list_end|>"; the model answers <|tool_call_start|>[name(arg="v", n=1)]<|tool_call_end|> (Python call
//           syntax); results go back as a `tool` turn. As in LiquidAI's chat template.
//   hermes  Qwen and the rest: "# Tools … <tools>{json per line}</tools> …" in the system turn; the model answers
//           <tool_call>{"name": …, "arguments": {…}}</tool_call>; results go back in <tool_response> in a user turn. As
//           in Qwen3's chat template.
// The parser also reads Qwen3-Coder's XML calls (<tool_call><function=name><parameter=k>v</parameter></function>), which
// LFM3 writes, whatever the format of the prompt.

const LFM2 = new Set(['lfm2', 'lfm2moe']);
export const toolFormat = (architecture) => (LFM2.has(architecture) ? 'lfm2' : 'hermes');

const fn = (t) => t.function || t;
const text = (c) => (typeof c === 'string' ? c : (c || []).filter((b) => b.type === 'text').map((b) => b.text).join('\n'));
const args = (a) => (typeof a === 'string' ? (a ? JSON.parse(a) : {}) : a || {});

// A Python literal for the lfm2 call syntax.
function py(v) {
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  if (Array.isArray(v)) return `[${v.map(py).join(', ')}]`;
  if (typeof v === 'object') return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${py(x)}`).join(', ')}}`;
  return JSON.stringify(v);
}

function callText(calls, format) {
  if (format === 'lfm2') {
    const list = calls.map((c) => `${fn(c).name}(${Object.entries(args(fn(c).arguments)).map(([k, v]) => `${k}=${py(v)}`).join(', ')})`);
    return `<|tool_call_start|>[${list.join(', ')}]<|tool_call_end|>`;
  }
  return calls.map((c) => `<tool_call>\n${JSON.stringify({ name: fn(c).name, arguments: args(fn(c).arguments) })}\n</tool_call>`).join('\n');
}

function systemText(tools, format) {
  const defs = tools.map(fn);
  if (format === 'lfm2') return `List of tools: <|tool_list_start|>${JSON.stringify(defs)}<|tool_list_end|>`;
  return '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function ' +
    `signatures within <tools></tools> XML tags:\n<tools>\n${defs.map((d) => JSON.stringify({ type: 'function', function: d })).join('\n')}\n</tools>\n\n` +
    'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n' +
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>';
}

// The conversation with the tool list in its system turn and tool traffic as text, for a back end whose chat template
// is not given the tools itself.
export function withTools(messages, tools, format) {
  const out = [];
  const list = tools?.length ? systemText(tools, format) : null;
  const first = messages[0]?.role === 'system' ? messages[0] : null;
  if (list) out.push({ role: 'system', content: first ? `${text(first.content)}\n${list}` : list });
  else if (first) out.push(first);
  for (const m of messages.slice(first ? 1 : 0)) {
    if (m.role === 'assistant' && m.tool_calls?.length) {
      const before = text(m.content);
      out.push({ role: 'assistant', content: `${before ? `${before}\n` : ''}${callText(m.tool_calls, format)}` });
    } else if (m.role === 'tool') {
      const body = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
      if (format === 'lfm2') out.push({ role: 'tool', content: body });
      else {
        const prev = out.at(-1);
        const wrapped = `<tool_response>\n${body}\n</tool_response>`;
        // Consecutive results share one user turn, as Qwen3's template writes them.
        if (prev?.role === 'user' && prev.toolResults) prev.content += `\n${wrapped}`;
        else out.push(Object.defineProperty({ role: 'user', content: wrapped }, 'toolResults', { value: true }));
      }
    } else out.push(m);
  }
  return out.map(({ role, content }) => ({ role, content }));
}

// A Python literal (strings, numbers, True/False/None, lists, dicts) at s[i]: [value, next index].
function literal(s, i) {
  const ws = () => { while (/\s/.test(s[i])) i++; };
  ws();
  const c = s[i];
  if (c === '"' || c === "'") {
    let out = '';
    for (i++; i < s.length && s[i] !== c; i++) {
      if (s[i] === '\\') { i++; out += ({ n: '\n', t: '\t', r: '\r' })[s[i]] ?? s[i]; } else out += s[i];
    }
    if (s[i] !== c) throw new Error('unterminated string');
    return [out, i + 1];
  }
  if (c === '[' || c === '(') {
    const close = c === '[' ? ']' : ')', arr = [];
    for (i++; ;) {
      ws();
      if (s[i] === close) return [arr, i + 1];
      let v; [v, i] = literal(s, i); arr.push(v); ws();
      if (s[i] === ',') i++;
      else if (s[i] !== close) throw new Error('bad list');
    }
  }
  if (c === '{') {
    const obj = {};
    for (i++; ;) {
      ws();
      if (s[i] === '}') return [obj, i + 1];
      let k, v; [k, i] = literal(s, i); ws();
      if (s[i++] !== ':') throw new Error('bad dict');
      [v, i] = literal(s, i); obj[k] = v; ws();
      if (s[i] === ',') i++;
      else if (s[i] !== '}') throw new Error('bad dict');
    }
  }
  const m = /^(True|False|None|true|false|null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)/.exec(s.slice(i));
  if (!m) throw new Error(`bad value at ${i}`);
  const t = m[1];
  return [t === 'True' || t === 'true' ? true : t === 'False' || t === 'false' ? false : t === 'None' || t === 'null' ? null : Number(t), i + t.length];
}

// [name(k=v, …), …] in Python call syntax.
function pythonCalls(body) {
  const s = body.trim().replace(/^\[/, '').replace(/\]$/, ''), calls = [];
  let i = 0;
  while (i < s.length) {
    const m = /^\s*,?\s*([A-Za-z_][\w.-]*)\s*\(/.exec(s.slice(i));
    if (!m) break;
    i += m[0].length;
    const a = {};
    for (;;) {
      while (/[\s,]/.test(s[i])) i++;
      if (s[i] === ')') { i++; break; }
      const k = /^([A-Za-z_]\w*)\s*=/.exec(s.slice(i));
      if (!k) throw new Error('bad argument');
      i += k[0].length;
      let v; [v, i] = literal(s, i); a[k[1]] = v;
    }
    calls.push({ name: m[1], arguments: a });
  }
  return calls;
}

function xmlCall(body) {
  const f = /<function=([^>\s]+)>([\s\S]*?)(?:<\/function>|$)/.exec(body);
  if (!f) return null;
  const a = {};
  for (const p of f[2].matchAll(/<parameter=([^>\s]+)>([\s\S]*?)<\/parameter>/g)) {
    const v = p[2].replace(/^\n/, '').replace(/\n$/, '');
    try { a[p[1]] = JSON.parse(v); } catch { a[p[1]] = v; }
  }
  return { name: f[1], arguments: a };
}

const PATTERNS = [
  [/<\|tool_call_start\|>([\s\S]*?)<\|tool_call_end\|>/g, (b) => pythonCalls(b)],
  [/<tool_call>([\s\S]*?)<\/tool_call>/g, (b) => {
    const x = xmlCall(b);
    if (x) return [x];
    const j = JSON.parse(b.trim());
    return [{ name: j.name, arguments: args(j.arguments ?? j.parameters) }];
  }],
];

// A bare LFM2 call list ending the reply, for an engine that decodes without special tokens (LocalMind's LFM2 engine
// drops <|tool_call_start|> and <|tool_call_end|>): "[name(k=v), …]" that parses whole as calls.
const BARE_LFM2 = /\[\s*[A-Za-z_][\w.-]*\s*\([\s\S]*\)\s*\]\s*$/;

// The reply's tool calls as OpenAI tool_calls, and its text without them. A call that does not parse stays in the
// text. ids are call_<n> in reply order. format 'lfm2' also reads a bare call list that ends the reply.
export function parseToolCalls(reply, format = null) {
  const calls = [];
  let content = reply;
  for (const [re, read] of PATTERNS) {
    content = content.replace(re, (whole, body) => {
      try { calls.push(...read(body)); return ''; } catch { return whole; }
    });
  }
  const bare = format === 'lfm2' && !calls.length && BARE_LFM2.exec(content);
  if (bare) {
    try {
      const found = pythonCalls(bare[0]);
      if (found.length) { calls.push(...found); content = content.slice(0, bare.index); }
    } catch { /* not a call list: leave the text */ }
  }
  return {
    content: calls.length ? content.trim() : reply,
    toolCalls: calls.map((c, n) => ({ id: `call_${n}`, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.arguments) } })),
  };
}
