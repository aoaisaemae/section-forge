import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';

const compiled = ts.transpileModule(fs.readFileSync(new URL('../app/waterfall.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
}).outputText;
const module = { exports: {} };
new Function('exports', 'module', compiled)(module.exports, module);
const { waterfall, waterfallCurve } = module.exports;
const fixture = (descending = false) => [[-.72, -.7], [0, -.2], [.72, .4]].map(([x, y]) => {
  const center = Array.from({ length: 15 }, (_, i) => [x + Math.sin(i / 14 * Math.PI) * .015, descending ? -y : y, -1 + i / 7]);
  return { center, left: center.map(p => [p[0] - .17, p[1], p[2]]), right: center.map(p => [p[0] + .17, p[1], p[2]]) };
});
for (const descending of [false, true]) for (const extension of [0, 55, 100]) for (const smoothing of [0, 70, 100]) {
  const ribbons = fixture(descending), { rows, rails } = waterfall(ribbons, extension, smoothing);
  assert.equal(rows.length, 96);
  assert(rows.every(row => row.length === rows[0].length), 'consistent section point count');
  for (const row of rows) {
    for (const [i, p] of row.entries()) {
      assert(p.every(Number.isFinite));
      assert(Math.abs(p[0]) <= 1.050000001 && Math.abs(p[2]) <= 1.050000001, '20ft footprint');
      if (i) {
        assert(p[0] >= row[i - 1][0] - 1e-10, 'no backwards oscillations between parallel rails');
        assert((descending ? -1 : 1) * (p[1] - row[i - 1][1]) >= -1e-10, 'monotone transition height');
      }
    }
    for (let i = 0; i <= 8; i++) assert.equal(row[i][1], row[0][1], 'flat floor band');
    assert(Math.abs(row[8][0] - row[0][0] - .34) < 1e-9, 'preserved circulation width');
  }
  assert.deepEqual(rails[0][0], ribbons[0].center[0]);
  assert.deepEqual(rails[0].at(-1), ribbons[0].center.at(-1));
}
const a = [-.5, -.7, 0], b = [.5, .4, 0], d = [1, 0, 0], h = .2, epsilon = 1e-5;
assert.deepEqual(waterfallCurve(a, b, d, d, h, h, 0), a);
assert.deepEqual(waterfallCurve(a, b, d, d, h, h, 1), b);
for (const t of [epsilon, 1 - epsilon]) {
  const p = waterfallCurve(a, b, d, d, h, h, t);
  assert(Math.abs(p[1] - (t < .5 ? a[1] : b[1])) < 1e-12, 'horizontal tangent and zero curvature at floor junction');
}
assert.deepEqual(waterfall([], 50, 50).rows, []);
assert.deepEqual(waterfall([{ center: [], left: [], right: [] }], 50, 50).rows, []);
assert.equal(waterfall([fixture()[0]], 50, 50).rows[0].length, 9);
console.log('Waterfall geometry checks passed.');
