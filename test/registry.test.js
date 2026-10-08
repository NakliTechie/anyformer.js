import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BACKENDS } from '../index.js';

test('default back ends: the ones that passed their browser tests, in preference order', () => {
  assert.deepEqual(BACKENDS.map((b) => b.id), ['transformers', 'kohra']);
});
