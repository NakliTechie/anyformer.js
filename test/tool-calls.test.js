import { test } from 'node:test';
import assert from 'node:assert/strict';
import { load } from '../index.js';
import { toolFormat, withTools, parseToolCalls } from '../src/tool-calls.js';

const weather = { type: 'function', function: { name: 'get_weather', description: 'Weather for a city', parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] } } };
const call = (name, a) => ({ name, arguments: a });
const calls = (r) => r.toolCalls.map((c) => call(c.function.name, JSON.parse(c.function.arguments)));

test('toolFormat: LFM2 architectures use the lfm2 format, the rest hermes', () => {
  assert.equal(toolFormat('lfm2'), 'lfm2');
  assert.equal(toolFormat('lfm2moe'), 'lfm2');
  assert.equal(toolFormat('qwen3'), 'hermes');
  assert.equal(toolFormat(undefined), 'hermes');
});

test('parseToolCalls: LFM2 Python call syntax, several calls, Python literals', () => {
  const r = parseToolCalls('Checking.<|tool_call_start|>[get_weather(city="Paris"), add(a=1, b=-2.5, deep={"k": [True, None, \'x\']})]<|tool_call_end|>');
  assert.equal(r.content, 'Checking.');
  assert.deepEqual(calls(r), [call('get_weather', { city: 'Paris' }), call('add', { a: 1, b: -2.5, deep: { k: [true, null, 'x'] } })]);
  assert.deepEqual(r.toolCalls.map((c) => [c.id, c.type]), [['call_0', 'function'], ['call_1', 'function']]);
});

test('parseToolCalls: hermes JSON and Qwen3-Coder XML; text without calls is unchanged', () => {
  assert.deepEqual(calls(parseToolCalls('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Pune"}}\n</tool_call>')), [call('get_weather', { city: 'Pune' })]);
  const xml = parseToolCalls('<tool_call>\n<function=get_weather>\n<parameter=city>\nLisbon\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>');
  assert.deepEqual(calls(xml), [call('get_weather', { city: 'Lisbon', days: 3 })]);
  assert.deepEqual(parseToolCalls('Plain answer.'), { content: 'Plain answer.', toolCalls: [] });
  // LFM2 without its markers (an engine that drops special tokens): read only for the lfm2 format, only at the end.
  const bare = parseToolCalls("I'll check.[get_weather(city=\"Paris\")]", 'lfm2');
  assert.equal(bare.content, "I'll check.");
  assert.deepEqual(calls(bare), [call('get_weather', { city: 'Paris' })]);
  assert.equal(parseToolCalls('[get_weather(city="Paris")]').toolCalls.length, 0);
  assert.equal(parseToolCalls('See [note(1)] above, then more text.', 'lfm2').toolCalls.length, 0);
  // An incomplete call (still streaming) stays text.
  assert.equal(parseToolCalls('<|tool_call_start|>[get_weather(city="Pa').toolCalls.length, 0);
});

test('withTools: lfm2 puts the list in the system turn and writes calls and results back in its format', () => {
  const msgs = withTools([
    { role: 'system', content: 'Be brief.' },
    { role: 'user', content: 'Weather in Paris?' },
    { role: 'assistant', content: null, tool_calls: [{ id: 'call_0', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] },
    { role: 'tool', tool_call_id: 'call_0', content: '{"temp": 18}' },
  ], [weather], 'lfm2');
  assert.equal(msgs[0].role, 'system');
  assert.match(msgs[0].content, /^Be brief\.\nList of tools: <\|tool_list_start\|>\[\{"name":"get_weather"/);
  assert.match(msgs[0].content, /<\|tool_list_end\|>$/);
  assert.equal(msgs[2].content, '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>');
  assert.deepEqual(msgs[3], { role: 'tool', content: '{"temp": 18}' });
  // What withTools writes, parseToolCalls reads back.
  assert.deepEqual(calls(parseToolCalls(msgs[2].content)), [call('get_weather', { city: 'Paris' })]);
});

test('withTools: hermes adds a system turn and folds consecutive results into one user turn', () => {
  const msgs = withTools([
    { role: 'user', content: 'Weather in Paris and Pune?' },
    { role: 'assistant', content: '', tool_calls: [
      { id: 'a', type: 'function', function: { name: 'get_weather', arguments: { city: 'Paris' } } },
      { id: 'b', type: 'function', function: { name: 'get_weather', arguments: { city: 'Pune' } } }] },
    { role: 'tool', content: '18' }, { role: 'tool', content: '31' },
  ], [weather], 'hermes');
  assert.match(msgs[0].content, /<tools>\n\{"type":"function","function":\{"name":"get_weather"/);
  assert.equal(msgs[2].content.match(/<tool_call>/g).length, 2);
  assert.deepEqual(msgs[3], { role: 'user', content: '<tool_response>\n18\n</tool_response>\n<tool_response>\n31\n</tool_response>' });
  assert.equal(msgs.length, 4);
});

const caps = { webgpu: true, f16: true };
function backend(reply, extra = {}) {
  const seen = [];
  return {
    seen,
    id: 't', canRun: () => ({ ok: true, estimate: { gpuBytes: 1, diskBytes: 0, tokPerSec: null } }),
    load: async () => ({ ...extra, async *generate(messages, o) { seen.push({ messages, o }); yield { text: reply, tokens: 3 }; }, dispose() {} }),
  };
}

test('load(): generate({ tools }) writes the tools into the prompt for the architecture and returns parsed toolCalls', async () => {
  const b = backend('<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>');
  const s = await load('a/b', { caps, model: { repo: 'a/b', formats: ['gguf'], architecture: 'lfm2moe' }, backends: [b] });
  const out = [];
  for await (const c of s.generate([{ role: 'user', content: 'Weather in Paris?' }], { tools: [weather] })) out.push(c);
  assert.match(b.seen[0].messages[0].content, /<\|tool_list_start\|>/);
  assert.equal(b.seen[0].o.tools, undefined);
  assert.equal(out[0].text, '');
  assert.deepEqual(calls(out[0]), [call('get_weather', { city: 'Paris' })]);
  // An assistant turn with only tool_calls is valid input.
  for await (const c of s.generate([{ role: 'user', content: 'x' }, { role: 'assistant', content: null, tool_calls: out[0].toolCalls }, { role: 'tool', content: '18' }], { tools: [weather] })) out.push(c);
  assert.equal(b.seen[1].messages.at(-1).role, 'tool');
});

test('load(): a back end whose template takes the tools gets them unchanged; the calls are still parsed', async () => {
  const b = backend('<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>', { templateTools: true });
  const s = await load('a/b', { caps, model: { repo: 'a/b', formats: ['onnx'], architecture: 'lfm3' }, backends: [b] });
  const msgs = [{ role: 'user', content: 'Weather in Paris?' }];
  let last;
  for await (const c of s.generate(msgs, { tools: [weather] })) last = c;
  assert.equal(b.seen[0].messages, msgs);
  assert.deepEqual(b.seen[0].o.tools, [weather]);
  assert.deepEqual(calls(last), [call('get_weather', { city: 'Paris' })]);
});
