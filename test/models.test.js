import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { MODELS, BACKENDS, plan, fromHubInfo } from '../index.js';

const entries = Object.values(MODELS).flatMap((family) => Object.values(family));
const caps = { webgpu: true, f16: true, subgroups: true, opfs: true };

test('every MODELS entry names a registered back end and an existing browser test page', () => {
  for (const m of entries) {
    assert.ok(BACKENDS.some((b) => b.id === m.backend), `${m.repo}: ${m.backend}`);
    assert.ok(existsSync(new URL(`./browser/${m.test}.html`, import.meta.url)), `${m.repo}: ${m.test}.html`);
    assert.ok(m.task && m.downloadBytes > 0, m.repo);
  }
  for (const family of Object.values(MODELS)) assert.ok(family.DEFAULT, 'each family has a DEFAULT');
});

test('an entry resolves to its own back end from its Hub fixture, with its task and budget', async () => {
  const fixtures = {
    'onnx-community/Qwen2.5-0.5B-Instruct': 'onnx-community_Qwen2.5-0.5B-Instruct',
    'onnx-community/Ternary-Bonsai-4B-ONNX': 'onnx-community_Ternary-Bonsai-4B-ONNX',
    'LiquidAI/LFM2.5-230M-GGUF': 'LiquidAI_LFM2.5-230M-GGUF',
    'LiquidAI/LFM2.5-1.2B-Instruct-GGUF': 'LiquidAI_LFM2.5-1.2B-Instruct-GGUF',
    'google/gemma-4-E2B-it-qat-mobile-transformers': 'google_gemma-4-E2B-it-qat-mobile-transformers',
    'google/gemma-4-E4B-it-qat-mobile-transformers': 'google_gemma-4-E4B-it-qat-mobile-transformers',
    'prism-ml/Ternary-Bonsai-2-27B-gguf': 'prism-ml_Ternary-Bonsai-2-27B-gguf',
    'naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX': 'naklitechie_Qwen3-0.6B-diffusion-mdlm-ONNX',
    'google/gemma-4-26B-A4B-it-qat-q4_0-gguf': 'google_gemma-4-26B-A4B-it-qat-q4_0-gguf',
    'Xenova/all-MiniLM-L6-v2': 'Xenova_all-MiniLM-L6-v2',
    'Xenova/detr-resnet-50': 'Xenova_detr-resnet-50',
  };
  for (const m of entries) {
    const model = fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${fixtures[m.repo]}.json`, import.meta.url))));
    const p = await plan(m, { model, caps });
    assert.equal(p.chosen, m.backend, m.repo);
    assert.equal(p.task, m.task, m.repo);
    if (m.budget) assert.equal(p.budget, m.budget);
  }
});

test('options passed with an entry win over the entry', async () => {
  const model = fromHubInfo(JSON.parse(readFileSync(new URL('./fixtures/google_gemma-4-26B-A4B-it-qat-q4_0-gguf.json', import.meta.url))));
  const p = await plan(MODELS.chat.GEMMA4_26B_A4B, { model, caps, budget: 7e9 });
  assert.equal(p.budget, 7e9);
});
