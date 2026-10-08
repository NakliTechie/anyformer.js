import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PINS, SOURCES, transformersBackend, kohraBackend, diskformerBackend, localmindBackend } from '../index.js';

test('every git pin is a full commit, transformers.js an exact version; nothing loads from a branch', () => {
  for (const k of ['kohra', 'diskformer', 'localmind']) assert.match(PINS[k], /^[0-9a-f]{40}$/, k);
  assert.match(PINS.transformers, /^\d+\.\d+\.\d+$/);
  assert.match(PINS.wllama, /^\d+\.\d+\.\d+$/);
  for (const url of Object.values(SOURCES)) assert.doesNotMatch(url, /@(main|master|latest)\b/, url);
});

test('each back end loads from its pinned source by default', () => {
  assert.equal(transformersBackend.source, SOURCES.transformers);
  assert.equal(kohraBackend.source, SOURCES.kohra);
  assert.equal(diskformerBackend.source, SOURCES.diskformer);
  assert.equal(localmindBackend.source, SOURCES.localmind);
  assert.ok(SOURCES.kohra.includes(PINS.kohra) && SOURCES.localmind.includes(PINS.localmind));
});
