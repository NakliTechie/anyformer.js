import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PINS, SOURCES, transformersBackend, kohraBackend, diskformerBackend, diskformerLlamaBackend, localmindBackend } from '../index.js';

test('every git pin is a full commit, transformers.js an exact version; nothing loads from a branch', () => {
  for (const k of ['kohra', 'diskformer']) assert.match(PINS[k], /^[0-9a-f]{40}$/, k);
  assert.equal(PINS.localmind, undefined, 'nothing loads from the LocalMind repo');
  assert.match(PINS.transformers, /^\d+\.\d+\.\d+$/);
  assert.match(PINS.wllama, /^\d+\.\d+\.\d+$/);
  for (const url of Object.values(SOURCES)) assert.doesNotMatch(url, /@(main|master|latest)\b/, url);
});

test('each back end loads from its pinned source by default', () => {
  assert.equal(transformersBackend.source, SOURCES.transformers);
  assert.equal(kohraBackend.source, SOURCES.kohra);
  assert.equal(diskformerBackend.source, SOURCES.diskformer);
  assert.equal(diskformerLlamaBackend.source, SOURCES.diskformerLlama);
  assert.ok(SOURCES.diskformerLlama.includes(PINS.diskformer) && SOURCES.diskformerLlama.endsWith('/ggml/llama.js'));
  assert.equal(localmindBackend.source, SOURCES.diskformer);         // LocalMind's engines, from diskformer's engines/
  assert.ok(SOURCES.kohra.includes(PINS.kohra));
  for (const url of Object.values(SOURCES)) assert.doesNotMatch(url, /NakliTechie\/LocalMind@/, url);
});
