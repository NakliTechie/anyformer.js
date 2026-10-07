import { test } from 'node:test';
import assert from 'node:assert/strict';
import { probe, memory64 } from '../index.js';

const fakeGpu = (features, info = {}) => ({
  requestAdapter: async () => ({
    features: new Set(features),
    limits: { maxBufferSize: 4294967296, maxStorageBufferBindingSize: 2147483648 },
    info,
  }),
});

test('no WebGPU, no storage: every field has its safe default', async () => {
  const caps = await probe({ gpu: undefined, storage: undefined });
  assert.deepEqual({ ...caps, memory64: undefined }, {
    webgpu: false, f16: false, subgroups: false, maxBufferSize: 0, maxStorageBufferBindingSize: 0, adapter: null,
    opfs: false, quotaBytes: 0, usageBytes: 0, memory64: undefined,
  });
  assert.equal(typeof caps.memory64, 'boolean');
});

test('adapter features, limits and info are read', async () => {
  const caps = await probe({ gpu: fakeGpu(['shader-f16', 'subgroups'], { vendor: 'apple', architecture: 'metal-3' }) });
  assert.equal(caps.webgpu, true);
  assert.equal(caps.f16, true);
  assert.equal(caps.subgroups, true);
  assert.equal(caps.maxBufferSize, 4294967296);
  assert.deepEqual(caps.adapter, { vendor: 'apple', architecture: 'metal-3', description: '' });
});

test('an adapter without shader-f16 reports f16 false', async () => {
  const caps = await probe({ gpu: fakeGpu([]) });
  assert.equal(caps.webgpu, true);
  assert.equal(caps.f16, false);
});

test('a null or rejected adapter request leaves webgpu false', async () => {
  assert.equal((await probe({ gpu: { requestAdapter: async () => null } })).webgpu, false);
  assert.equal((await probe({ gpu: { requestAdapter: async () => { throw new Error('x'); } } })).webgpu, false);
});

test('OPFS and the storage estimate are read; a failed estimate gives 0', async () => {
  const caps = await probe({ storage: { getDirectory() {}, estimate: async () => ({ quota: 1e11, usage: 5e9 }) } });
  assert.equal(caps.opfs, true);
  assert.equal(caps.quotaBytes, 1e11);
  assert.equal(caps.usageBytes, 5e9);
  const bad = await probe({ storage: { getDirectory() {}, estimate: async () => { throw new Error('x'); } } });
  assert.equal(bad.opfs, true);
  assert.equal(bad.quotaBytes, 0);
});

test('memory64 probe agrees with a direct validate of the same bytes', () => {
  const bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 5, 3, 1, 4, 0]);
  assert.equal(memory64(), WebAssembly.validate(bytes));
});
