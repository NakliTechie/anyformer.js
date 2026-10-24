// The agent face: every public anyformer call as a tool (name, description, JSON-schema input, handler), published
// through WebMCP (navigator.modelContext) where the browser has it and as window.anyformer.tools otherwise. Tests drive
// this manifest, not the functions behind it.
//
// Consequence classes: 'read' runs freely; 'stages' lands only when the call accepts the consequence, otherwise it
// returns what would happen: anyformer_load (downloads weights, fills the GPU) needs acceptDiskBytes, anyformer_remove
// (deletes a downloaded model; the next load downloads it again) needs confirm.
// Person-only acts: none. anyformer has no act whose point is the person's own authorship or consent.

import { probe } from './probe.js';
import { plan, load } from './load.js';
import { MODELS } from './models.js';
import { stored, remove } from './store.js';

// Options every tool call passes to plan() and load(): a page sets backends or budget here; tests inject caps and fetch.
export const toolOptions = {};

const sessions = new Map();
let nextId = 1;
// The trajectory: the last 50 calls, with the door each came through. Bounded, so it never grows with use.
export const journal = [];
const record = (entry) => { journal.push({ at: new Date().toISOString(), ...entry }); if (journal.length > 50) journal.shift(); };

const repo = { type: 'string', description: 'Hugging Face repo id, e.g. onnx-community/Qwen2.5-0.5B-Instruct' };
const budget = { type: 'number', description: 'GPU memory budget in bytes; default 2.5e9' };

// A plan without the file list: bounded by the number of back ends, not the size of the repo.
function brief(p) {
  return {
    repo: p.model.repo, formats: p.model.formats, architecture: p.model.architecture, task: p.model.task,
    budget: p.budget, chosen: p.chosen, ranked: p.ranked,
    candidates: p.candidates.map(({ order, ...c }) => c),
    diskBytes: p.candidates.find((c) => c.backend === p.chosen)?.estimate?.diskBytes ?? null,
  };
}

export const TOOLS = [
  {
    name: 'anyformer_probe', consequence: 'read',
    description: 'What this browser can run: WebGPU, shader-f16, subgroups, buffer limits, adapter, OPFS quota, memory64.',
    inputSchema: { type: 'object', properties: {} },
    handler: () => probe(),
  },
  {
    name: 'anyformer_models', consequence: 'read',
    description: 'Models that pass a browser test through anyformer, by family (chat, embedding, objectDetection): repo, back end, task, budget, download size, and the flags diffusion and experimental. Pass a repo to anyformer_plan or anyformer_load.',
    inputSchema: { type: 'object', properties: {} },
    handler: () => MODELS,
  },
  {
    name: 'anyformer_plan', consequence: 'read',
    description: 'Which back end would run a model under a GPU budget, and why every other back end refused (closed codes: format, model, webgpu, f16, opfs, budget, threw). Loads nothing.',
    inputSchema: { type: 'object', properties: { repo, budget }, required: ['repo'] },
    handler: async ({ repo, budget }) => brief(await plan(repo, { ...toolOptions, ...(budget && { budget }) })),
  },
  {
    name: 'anyformer_load', consequence: 'stages',
    description: 'Load a model. Downloads its weights to browser storage on first use. Pass acceptDiskBytes >= the plan\'s diskBytes to load; without it the call returns the plan and loads nothing.',
    inputSchema: {
      type: 'object', required: ['repo'],
      properties: { repo, budget, acceptDiskBytes: { type: 'number', description: 'disk bytes the caller accepts this load may write' } },
    },
    handler: async ({ repo, budget, acceptDiskBytes }) => {
      const p = brief(await plan(repo, { ...toolOptions, ...(budget && { budget }) }));
      if (!p.chosen) return { loaded: false, plan: p };
      if (!(acceptDiskBytes >= (p.diskBytes || 0))) return { loaded: false, staged: true, plan: p };
      const s = await load(repo, { ...toolOptions, ...(budget && { budget }) });
      const session = `s${nextId++}`;
      sessions.set(session, { repo, backend: s.backend, s });
      return { loaded: true, session, backend: s.backend };
    },
  },
  {
    name: 'anyformer_generate', consequence: 'read',
    description: 'Reply to a chat with a loaded model. Greedy decoding; returns the whole reply.',
    inputSchema: {
      type: 'object', required: ['session', 'messages'],
      properties: {
        session: { type: 'string' },
        messages: { type: 'array', items: { type: 'object', required: ['role', 'content'], properties: { role: { type: 'string' }, content: { type: 'string' } } } },
        maxNewTokens: { type: 'number', description: 'default 256' },
      },
    },
    handler: async ({ session, messages, maxNewTokens = 256 }) => {
      const e = sessions.get(session);
      if (!e) throw new Error(`no session ${session}; anyformer_sessions lists the open ones, anyformer_load opens one`);
      let last = { text: '', tokens: 0 };
      for await (const c of e.s.generate(messages, { maxNewTokens })) last = c;
      return { text: last.text, tokens: last.tokens };
    },
  },
  {
    name: 'anyformer_sessions', consequence: 'read',
    description: 'The loaded models, and the last 50 tool calls with the door each came through.',
    inputSchema: { type: 'object', properties: {} },
    handler: () => ({ sessions: [...sessions].map(([id, e]) => ({ session: id, repo: e.repo, backend: e.backend })), journal }),
  },
  {
    name: 'anyformer_stored', consequence: 'read',
    description: 'Models downloaded into this browser: back end, repo, bytes.',
    inputSchema: { type: 'object', properties: {} },
    handler: () => stored(),
  },
  {
    name: 'anyformer_remove', consequence: 'stages',
    description: 'Delete a downloaded model from this browser. Without confirm: true it returns what it would delete and deletes nothing. The next load downloads it again.',
    inputSchema: { type: 'object', required: ['repo'], properties: { repo, confirm: { type: 'boolean' } } },
    handler: async ({ repo, confirm }) => {
      if (confirm !== true) return { removed: false, staged: true, would: (await stored()).filter((r) => r.repo === repo) };
      return { removed: true, ...(await remove(repo)) };
    },
  },
  {
    name: 'anyformer_unload', consequence: 'read',
    description: 'Free a loaded model\'s GPU memory. Its weights stay in browser storage for the next load.',
    inputSchema: { type: 'object', required: ['session'], properties: { session: { type: 'string' } } },
    handler: async ({ session }) => {
      const e = sessions.get(session);
      if (!e) return { unloaded: false };
      sessions.delete(session);
      await e.s.dispose();
      return { unloaded: true };
    },
  },
];

// Call a tool by name, as either door does. Errors come back as { error } so a caller branches on data, not a throw.
export async function call(name, input = {}, door = 'direct') {
  const t = TOOLS.find((x) => x.name === name);
  if (!t) return { error: `no tool ${name}; tools: ${TOOLS.map((x) => x.name).join(', ')}` };
  try {
    const out = await t.handler(input);
    record({ tool: name, door, ok: true });
    return out;
  } catch (e) {
    record({ tool: name, door, ok: false, error: e.message });
    return { error: e.message, ...(e.code && { code: e.code }), ...(e.need && { need: e.need }), ...(e.plan && { plan: brief(e.plan) }) };
  }
}

// Publish the manifest: WebMCP when present, window.anyformer.tools always (the same entries).
export function registerTools({ modelContext = globalThis.navigator?.modelContext, target = globalThis } = {}) {
  if (modelContext?.registerTool)
    for (const t of TOOLS)
      modelContext.registerTool({ name: t.name, description: t.description, inputSchema: t.inputSchema, execute: (input) => call(t.name, input, 'modelContext') });
  target.anyformer = { ...(target.anyformer || {}), tools: Object.fromEntries(TOOLS.map((t) => [t.name, (input) => call(t.name, input, 'window')])) };
  return TOOLS.map((t) => t.name);
}
