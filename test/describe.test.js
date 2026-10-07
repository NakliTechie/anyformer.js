import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, fromHubInfo } from '../index.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url)));
const gemma = fixture('google_gemma-4-26B-A4B-it-qat-q4_0-gguf');
const qwen = fixture('onnx-community_Qwen2.5-0.5B-Instruct');
const kohra = fixture('naklitechie_Qwen3-0.6B-diffusion-mdlm-ONNX');

test('GGUF repo: architecture from the gguf block, both gguf files, total bytes', () => {
  const d = fromHubInfo(gemma);
  assert.equal(d.repo, 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf');
  assert.equal(d.revision, 'd1c082be9cf3c8a514acf63b8761f4b41935842e');
  assert.deepEqual(d.formats, ['gguf']);
  assert.equal(d.architecture, 'gemma4');
  assert.equal(d.task, 'image-text-to-text');
  assert.deepEqual(d.files.gguf.map((f) => f.name).sort(), ['gemma-4-26B-it-mmproj.gguf', 'gemma-4-26B_q4_0-it.gguf']);
  assert.equal(d.files.gguf.find((f) => f.name === 'gemma-4-26B_q4_0-it.gguf').size, 14439363584);
  assert.equal(d.ggufTotalBytes, 25233142046);
});

test('ONNX repo: architecture from config.model_type, 8 onnx files, no gguf total', () => {
  const d = fromHubInfo(qwen);
  assert.deepEqual(d.formats, ['onnx']);
  assert.equal(d.architecture, 'qwen2');
  assert.equal(d.task, 'text-generation');
  assert.equal(d.files.onnx.length, 8);
  assert.equal(d.files.gguf.length, 0);
  assert.equal(d.ggufTotalBytes, null);
});

test('kohra repo: onnx format, text-diffusion tag, fp16 and q4 graphs', () => {
  const d = fromHubInfo(kohra);
  assert.deepEqual(d.formats, ['onnx']);
  assert.ok(d.tags.includes('text-diffusion'));
  assert.ok(d.tags.includes('kohra'));
  assert.deepEqual(d.files.onnx.map((f) => f.name).sort(), ['onnx/model_fp16_fused.onnx', 'onnx/model_q4f16_rtn_sym.onnx']);
});

test('empty Hub JSON gives null fields, not a throw', () => {
  const d = fromHubInfo({ id: 'x/y' });
  assert.deepEqual(d.formats, []);
  assert.equal(d.architecture, null);
  assert.equal(d.revision, null);
  assert.deepEqual(d.tags, []);
});

test('describe() builds the Hub URL, pins a revision, and reports HTTP errors', async () => {
  const seen = [];
  const fetch = async (url) => { seen.push(url); return { ok: true, json: async () => qwen }; };
  const d = await describe('onnx-community/Qwen2.5-0.5B-Instruct', { fetch });
  assert.equal(d.architecture, 'qwen2');
  await describe('a/b', { fetch, revision: 'abc123', endpoint: 'https://mirror.test' });
  assert.deepEqual(seen, [
    'https://huggingface.co/api/models/onnx-community/Qwen2.5-0.5B-Instruct?blobs=true',
    'https://mirror.test/api/models/a/b/revision/abc123?blobs=true',
  ]);
  const notFound = async () => ({ ok: false, status: 404 });
  await assert.rejects(describe('a/missing', { fetch: notFound }), /returned 404/);
});
