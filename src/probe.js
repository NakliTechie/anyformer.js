// What this browser can do, read once per page. Every field has a safe value when an API is missing (Node, Firefox,
// a page without WebGPU), so a back end's canRun() reads caps without guarding.

// A wasm module with one memory whose flags byte is 0x04 (memory64). It validates only where memory64 is supported.
const MEMORY64_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 5, 3, 1, 4, 0]);

export function memory64() {
  try { return WebAssembly.validate(MEMORY64_PROBE); } catch { return false; }
}

export async function probe({ gpu = globalThis.navigator?.gpu, storage = globalThis.navigator?.storage } = {}) {
  const caps = {
    webgpu: false, f16: false, subgroups: false,
    maxBufferSize: 0, maxStorageBufferBindingSize: 0, adapter: null,
    opfs: false, quotaBytes: 0, usageBytes: 0,
    memory64: memory64(),
  };
  const adapter = gpu ? await gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null) : null;
  if (adapter) {
    caps.webgpu = true;
    caps.f16 = adapter.features.has('shader-f16');
    caps.subgroups = adapter.features.has('subgroups');
    caps.maxBufferSize = adapter.limits.maxBufferSize;
    caps.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    const info = adapter.info || {};
    caps.adapter = { vendor: info.vendor || '', architecture: info.architecture || '', description: info.description || '' };
  }
  if (storage?.getDirectory) {
    caps.opfs = true;
    const est = storage.estimate ? await storage.estimate().catch(() => null) : null;
    caps.quotaBytes = est?.quota || 0;
    caps.usageBytes = est?.usage || 0;
  }
  return caps;
}
