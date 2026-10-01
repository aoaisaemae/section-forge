type V = [number, number, number];
type Ribbon = { center: V[]; left: V[]; right: V[] };
const HALF = 1.05;
const mix = (a: V, b: V, t: number): V => a.map((v, i) => v + (b[i] - v) * t) as V;
const move = (p: V, direction: V, distance: number): V => p.map((v, i) => v + direction[i] * distance) as V;
const distance = (a: V, b: V) => Math.hypot(...a.map((v, i) => v - b[i]));

// Six controls give horizontal first derivatives and zero second derivatives
// at the floor junctions. The height is monotone, without extra oscillations.
export function waterfallCurve(a: V, b: V, da: V, db: V, ha: number, hb: number, t: number): V {
  let controls = [a, move(a, da, ha), move(a, da, 2 * ha), move(b, db, -2 * hb), move(b, db, -hb), b];
  while (controls.length > 1) controls = controls.slice(1).map((p, i) => mix(controls[i], p, t));
  return controls[0];
}

function room(p: V, d: V) {
  return Math.max(0, Math.min(...[0, 2].map(k => Math.abs(d[k]) < 1e-9 ? Infinity : (Math.sign(d[k]) * HALF - p[k]) / d[k])));
}
function sample(line: V[], count: number): V[] {
  return Array.from({ length: count }, (_, i) => {
    const u = i / (count - 1) * (line.length - 1), a = Math.floor(u);
    return mix(line[a], line[Math.min(a + 1, line.length - 1)], u - a);
  });
}

export function waterfall(input: Ribbon[], extension: number, smoothing: number) {
  const ribbons = input.filter(r => r.center.length > 1 && r.left.length > 1 && r.right.length > 1);
  if (!ribbons.length) return { rows: [] as V[][], rails: [] as V[][] };
  const count = 96;
  const rails = ribbons.map(r => sample(r.center, count));
  let widths = ribbons.map(r => {
    const left = sample(r.left, count), right = sample(r.right, count);
    return left.map((p, i) => [p[0] - right[i][0], 0, p[2] - right[i][2]] as V);
  });
  // Smooth corresponding longitudinal stations; never reorder points within a rail.
  for (let pass = 0; pass < Math.round(smoothing / 10); pass++) {
    widths = widths.map(old => old.map((w, i) => {
      if (i === 0 || i === count - 1) return w;
      const average = mix(w, mix(old[i - 1], old[i + 1], .5), .4), length = Math.hypot(average[0], average[2]);
      const original = Math.hypot(w[0], w[2]);
      return length > 1e-9 ? [average[0] * original / length, 0, average[2] * original / length] as V : w;
    }));
    rails.forEach((old, r) => {
      rails[r] = old.map((p, i) => i === 0 || i === count - 1 ? p : mix(p, mix(old[i - 1], old[i + 1], .5), .4));
    });
  }
  // One proximity order for every cross-section avoids station-by-station twists.
  const remaining = rails.map((_, i) => i), order = [remaining.shift()!];
  const cost = (a: number, b: number) => rails[a].reduce((sum, p, i) => sum + distance(p, rails[b][i]), 0);
  while (remaining.length) {
    const last = order[order.length - 1];
    remaining.sort((a, b) => cost(last, a) - cost(last, b));
    order.push(remaining.shift()!);
  }
  // Orient each original offset pair once, towards its neighboring circulation.
  const signs = order.map((r, i) => {
    if (order.length === 1) return 1;
    const middle = Math.floor(count / 2), previous = order[Math.max(0, i - 1)], next = order[Math.min(order.length - 1, i + 1)];
    const delta = rails[next][middle].map((v, k) => v - rails[previous][middle][k]);
    return widths[r][middle][0] * delta[0] + widths[r][middle][2] * delta[2] < 0 ? -1 : 1;
  });
  const rows = Array.from({ length: count }, (_, station) => {
    const edges = order.map((r, i) => {
      const p = rails[r][station], w = widths[r][station], half = Math.hypot(w[0], w[2]) / 2;
      const direction: V = half > 1e-9 ? [signs[i] * w[0] / (2 * half), 0, signs[i] * w[2] / (2 * half)] : [1, 0, 0];
      // Limit the whole offset, rather than clamping individual curve points.
      const inward: V = [-direction[0], 0, -direction[2]];
      return { left: move(p, inward, Math.min(half, room(p, inward))), right: move(p, direction, Math.min(half, room(p, direction))), direction };
    });
    const row: V[] = [];
    edges.forEach((edge, i) => {
      for (let j = i === 0 ? 0 : 1; j <= 8; j++) row.push(mix(edge.left, edge.right, j / 8));
      if (i === edges.length - 1) return;
      const next = edges[i + 1], gap = Math.hypot(edge.right[0] - next.left[0], edge.right[2] - next.left[2]);
      const forward = edge.direction, backward: V = [-next.direction[0], 0, -next.direction[2]];
      const pad = gap * .22 * Math.max(0, Math.min(1, extension / 100));
      const a = move(edge.right, forward, Math.min(pad, room(edge.right, forward) * .45));
      const b = move(next.left, backward, Math.min(pad, room(next.left, backward) * .45));
      for (let j = 1; j <= 4; j++) row.push(mix(edge.right, a, j / 4));
      const handle = Math.hypot(a[0] - b[0], a[2] - b[2]) * .22;
      const ha = Math.min(handle, room(a, forward) / 2), hb = Math.min(handle, room(b, backward) / 2);
      for (let j = 1; j <= 48; j++) row.push(waterfallCurve(a, b, forward, next.direction, ha, hb, j / 48));
      for (let j = 1; j <= 4; j++) row.push(mix(b, next.left, j / 4));
    });
    return row;
  });
  return { rows, rails };
}
