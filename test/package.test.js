import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

// What npm would publish: the code and its docs, never tests, benches, plans or examples.
test('the package ships index.js, every src module, README, LICENSE and SPEC, and nothing else', () => {
  const [pack] = JSON.parse(execFileSync('npm', ['pack', '--dry-run', '--json'], { cwd: new URL('..', import.meta.url), encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }));
  const files = pack.files.map((f) => f.path).sort();
  for (const f of ['index.js', 'src/load.js', 'src/resolve.js', 'src/sources.js', 'src/tools.js', 'src/backends/localmind.js', 'README.md', 'LICENSE', 'SPEC.md', 'package.json'])
    assert.ok(files.includes(f), f);
  for (const f of files) assert.doesNotMatch(f, /^(test|bench|plan|examples|scripts)\//, f);
});
