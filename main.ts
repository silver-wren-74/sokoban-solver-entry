// Sokoban solver: push-optimal A* over pushes, room by room.
//
// A level from this generator is a hallway with independent rooms off it:
// a box never leaves its room, so a level's push optimum is the sum of its
// rooms' optima. The solver cuts the level into rooms (the dead squares of
// the player's row separate them), solves each room with an A* search over
// pushes — a state is the boxes plus the player's reachable region — and
// stitches the rooms' move strings together, the player walking on from
// where the last room left it. A room the decomposition cannot isolate is
// handled by the same search over the whole level.
//
// The A* estimate is a minimum-cost matching of boxes to goals over true
// single-box push distances (a box's pushes counted with the other boxes
// ignored), which never overestimates, so the first solution found is
// push-optimal. Boxes standing on dead squares (squares from which no push
// reaches a goal) prune a state at once, and so does a push that completes
// a frozen 2x2 block of walls and boxes with a box off its goal. All
// budgets are counted in
// expanded states, never read from a clock, so two runs write the same
// bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in a fixed order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const WALK_LETTER = ['u', 'd', 'l', 'r'];
const PUSH_LETTER = ['U', 'D', 'L', 'R'];
const OPPOSITE = [1, 0, 3, 2];

// Work budgets, in expanded A* states: generous per room, bounded overall,
// so the whole set always finishes well inside its wall budget.
const ROOM_BUDGET = 1_200_000;
const LEVEL_BUDGET = 2_500_000;
const SET_BUDGET = 9_000_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text; squares outside the rows and on the outer frame are walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** Neighbour table: nbr[i * 4 + d] is the square one step in direction d, or -1 off the board. */
function neighbours(level: Level): Int32Array {
  const { w, h } = level;
  const nbr = new Int32Array(w * h * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (nx >= 0 && ny >= 0 && nx < w && ny < h) nbr[i * 4 + d] = ny * w + nx;
      }
    }
  }
  return nbr;
}

/**
 * Single-box push distances to each goal: dist[g][s] is the fewest pushes
 * that bring a box from square s to goal g with every other box ignored
 * (a pull-back breadth-first search from the goal), or -1 when impossible.
 */
function goalDistances(level: Level, nbr: Int32Array): Int32Array[] {
  const n = level.w * level.h;
  return level.goals.map((g) => {
    const dist = new Int32Array(n).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let head = 0; head < queue.length; head++) {
      const x = queue[head];
      for (let d = 0; d < 4; d++) {
        // The box was pushed in direction d from y into x; the player stood at y - step(d).
        const y = nbr[x * 4 + OPPOSITE[d]];
        if (y < 0) continue;
        const stand = nbr[y * 4 + OPPOSITE[d]];
        if (stand < 0) continue;
        if (level.wall[y] || level.wall[stand] || dist[y] !== -1) continue;
        dist[y] = dist[x] + 1;
        queue.push(y);
      }
    }
    return dist;
  });
}

/**
 * The A* estimate for a box placement: the cheapest assignment of boxes to
 * distinct goals under the single-box push distances (subset DP for up to
 * nine boxes, a sum of per-box minima beyond that). Infinity when some box
 * reaches no goal at all: such a state can never be solved.
 */
function estimate(boxes: number[], dists: Int32Array[]): number {
  const nb = boxes.length;
  const ng = dists.length;
  if (nb > 9) {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (let g = 0; g < ng; g++) {
        const d = dists[g][b];
        if (d >= 0 && d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  let dp = new Float64Array(1 << ng).fill(Infinity);
  dp[0] = 0;
  for (let i = 0; i < nb; i++) {
    const next = new Float64Array(1 << ng).fill(Infinity);
    const b = boxes[i];
    for (let mask = 0; mask < dp.length; mask++) {
      const cur = dp[mask];
      if (cur === Infinity) continue;
      for (let g = 0; g < ng; g++) {
        if (mask & (1 << g)) continue;
        const d = dists[g][b];
        if (d < 0) continue;
        const nm = mask | (1 << g);
        if (cur + d < next[nm]) next[nm] = cur + d;
      }
    }
    dp = next;
  }
  let best = Infinity;
  for (const v of dp) if (v < best) best = v;
  return best;
}

/** Flood fill: mark the squares the player reaches from `from`; return the region's lowest square. */
function flood(
  level: Level,
  nbr: Int32Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let top = 0;
  stack[top++] = from;
  seen[from] = stamp;
  while (top > 0) {
    const p = stack[--top];
    if (p < low) low = p;
    for (let d = 0; d < 4; d++) {
      const q = nbr[p * 4 + d];
      if (q >= 0 && !level.wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[top++] = q;
      }
    }
  }
  return low;
}

interface Solution {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal A* for one (sub-)level: the player's start square, the boxes
 * and the goals are the level's own; everything outside is already walled
 * off by the caller. Expands at most `budget` states (counted through the
 * shared `spent` cell) and returns null when the budget runs out.
 */
function solveOptimal(level: Level, budget: number, spent: { n: number }): Solution | null {
  const nbr = neighbours(level);
  const dists = goalDistances(level, nbr);
  const n = level.w * level.h;
  const goalSet = new Uint8Array(n);
  for (const g of level.goals) goalSet[g] = 1;
  // A square is live when some goal's distance table reaches it.
  const live = new Uint8Array(n);
  for (const dist of dists) {
    for (let i = 0; i < n; i++) if (dist[i] >= 0) live[i] = 1;
  }
  if (level.boxes.some((b) => !live[b])) return null;
  const h0 = estimate(level.boxes, dists);
  if (h0 === Infinity) return null;

  const occupied = new Uint8Array(n);
  // Two stamp boards: `seen` holds the region of the state being expanded
  // while children are tested against it, `mark` the regions of the
  // children those pushes make — one board would overwrite the other.
  const seen = new Int32Array(n);
  const mark = new Int32Array(n);
  const stack = new Int32Array(n);
  let stamp = 0;

  // State store: boxes, the square the player stands on, g, parent, and the
  // push that made the state (the box's square before the push, its direction).
  const stBoxes: number[][] = [level.boxes.slice()];
  const stPlayer: number[] = [level.player];
  const stG: number[] = [0];
  const stParent: number[] = [-1];
  const stBoxFrom: number[] = [-1];
  const stDir: number[] = [-1];
  const stKey: string[] = [''];
  const bestG = new Map<string, number>();

  for (const b of level.boxes) occupied[b] = 1;
  const low0 = flood(level, nbr, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  stKey[0] = `${low0}|${level.boxes.join(',')}`;
  bestG.set(stKey[0], 0);

  // Binary heap of state indices ordered by f = g + h, ties to the deeper
  // state, then to the earlier one: fully deterministic.
  const heapIdx: number[] = [0];
  const heapF: number[] = [h0];
  const before = (a: number, b: number): boolean => {
    if (heapF[a] !== heapF[b]) return heapF[a] < heapF[b];
    if (stG[heapIdx[a]] !== stG[heapIdx[b]]) return stG[heapIdx[a]] > stG[heapIdx[b]];
    return heapIdx[a] < heapIdx[b];
  };
  const heapPush = (idx: number, f: number): void => {
    heapIdx.push(idx);
    heapF.push(f);
    let i = heapIdx.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(i, p)) break;
      [heapIdx[i], heapIdx[p]] = [heapIdx[p], heapIdx[i]];
      [heapF[i], heapF[p]] = [heapF[p], heapF[i]];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heapIdx[0];
    const lastIdx = heapIdx.pop() as number;
    const lastF = heapF.pop() as number;
    if (heapIdx.length > 0) {
      heapIdx[0] = lastIdx;
      heapF[0] = lastF;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heapIdx.length) break;
        if (c + 1 < heapIdx.length && before(c + 1, c)) c++;
        if (!before(c, i)) break;
        [heapIdx[i], heapIdx[c]] = [heapIdx[c], heapIdx[i]];
        [heapF[i], heapF[c]] = [heapF[c], heapF[i]];
        i = c;
      }
    }
    return top;
  };

  // Whether the box just pushed to `t` completes a frozen 2x2 block of
  // walls and boxes with some box off its goal: that box can never move.
  const frozenBlock = (t: number): boolean => {
    const w = level.w;
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const cells = [a, a + 1, a + w, a + w + 1];
      if (!cells.every((c) => level.wall[c] || occupied[c])) continue;
      if (cells.some((c) => occupied[c] && !goalSet[c])) return true;
    }
    return false;
  };

  let found = -1;
  while (heapIdx.length > 0) {
    const s = heapPop();
    const boxes = stBoxes[s];
    const g = stG[s];
    // A stale heap entry: this state was re-reached more cheaply since.
    if (bestG.get(stKey[s]) !== g) continue;
    if (boxes.every((b) => goalSet[b])) {
      found = s;
      break;
    }
    if (spent.n >= budget) return null;
    spent.n++;
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    flood(level, nbr, occupied, stPlayer[s], seen, here, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const stand = nbr[b * 4 + OPPOSITE[d]];
        const target = nbr[b * 4 + d];
        if (stand < 0 || target < 0) continue;
        if (seen[stand] !== here || level.wall[target] || occupied[target] || !live[target]) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenBlock(target);
        occupied[target] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const next = boxes.slice();
        next[bi] = target;
        next.sort((a, b2) => a - b2);
        // Region of the new state: the player ends on the box's old square.
        occupied[b] = 0;
        occupied[target] = 1;
        const low = flood(level, nbr, occupied, b, mark, ++stamp, stack);
        occupied[target] = 0;
        occupied[b] = 1;
        const key = `${low}|${next.join(',')}`;
        const ng = g + 1;
        const prev = bestG.get(key);
        if (prev !== undefined && prev <= ng) continue;
        const h = estimate(next, dists);
        if (h === Infinity) continue;
        bestG.set(key, ng);
        const idx = stBoxes.length;
        stBoxes.push(next);
        stPlayer.push(b);
        stG.push(ng);
        stParent.push(s);
        stBoxFrom.push(b);
        stDir.push(d);
        stKey.push(key);
        heapPush(idx, ng + h);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the push sequence, then spell it out with the walks between pushes.
  const chain: number[] = [];
  for (let s = found; s > 0; s = stParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = level.player;
  const cur = level.boxes.slice();
  const occ = new Uint8Array(n);
  for (const s of chain) {
    const from = stBoxFrom[s];
    const d = stDir[s];
    const stand = nbr[from * 4 + OPPOSITE[d]];
    for (const b of cur) occ[b] = 1;
    moves += walkTo(level, nbr, occ, player, stand);
    for (const b of cur) occ[b] = 0;
    moves += PUSH_LETTER[d];
    cur[cur.indexOf(from)] = nbr[from * 4 + d];
    cur.sort((a, b) => a - b);
    player = from;
  }
  return { pushes: chain.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(level: Level, nbr: Int32Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const n = level.w * level.h;
  const prev = new Int32Array(n).fill(-1);
  const how = new Int32Array(n).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let head = 0; head < queue.length; head++) {
    const p = queue[head];
    if (p === to) break;
    for (let d = 0; d < 4; d++) {
      const q = nbr[p * 4 + d];
      if (q < 0 || level.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = WALK_LETTER[how[p]] + s;
  return s;
}

/** The square the player stands on after playing `moves` from `from`. */
function endOf(level: Level, from: number, moves: string): number {
  const nbr = neighbours(level);
  let p = from;
  for (const ch of moves.toLowerCase()) {
    const d = WALK_LETTER.indexOf(ch);
    p = nbr[p * 4 + d];
  }
  return p;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor; each part left that holds
 * boxes is a room. Null when the level is no clean set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const nbr = neighbours(level);
  const dists = goalDistances(level, nbr);
  const n = level.w * level.h;
  const live = new Uint8Array(n);
  for (const dist of dists) for (let i = 0; i < n; i++) if (dist[i] >= 0) live[i] = 1;
  const row = Math.floor(level.player / level.w);
  const cut = new Uint8Array(n);
  for (let i = row * level.w; i < (row + 1) * level.w; i++) {
    if (!level.wall[i] && !live[i]) cut[i] = 1;
  }
  if (level.boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(n).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < n; s++) {
    if (level.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let d = 0; d < 4; d++) {
        const q = nbr[squares[i] * 4 + d];
        if (q >= 0 && !level.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of level.boxes) rooms[part[b]].boxes.push(b);
  for (const g of level.goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (level.goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one level: room by room when it decomposes, whole-level otherwise. */
function solveLevel(level: Level, spent: { n: number }): Solution | null {
  const rooms = roomsOf(level);
  if (rooms === null) {
    return solveOptimal(level, spent.n + LEVEL_BUDGET, spent);
  }
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice(),
      boxes: room.boxes.slice(),
      player,
    };
    const solved = solveOptimal(sub, spent.n + ROOM_BUDGET, spent);
    if (solved === null) {
      // One room resisted its budget: try the whole level with what is left.
      return solveOptimal(level, spent.n + LEVEL_BUDGET, spent);
    }
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

function solve(text: string, spent: { n: number }): string {
  const level = parseLevel(text);
  if (level.player < 0 || level.boxes.length === 0) return '';
  const solved = solveLevel(level, spent);
  return solved === null ? '' : solved.moves;
}

const spent = { n: 0 };
for (const id of readdirSync(INPUT_DIR).sort()) {
  if (spent.n >= SET_BUDGET) {
    // Out of work budget: still answer in format, with no moves.
    mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
    writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), '\n');
    continue;
  }
  const moves = solve(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'), spent);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
