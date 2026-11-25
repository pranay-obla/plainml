// Builds the home page's held-out figure: a card sorter on a bench plate. A pusher feeds row cards from
// a hopper down a raised channel; a flap sends every fifth card into a locked drawer (the test rows) and
// the rest on to the training tray. 30 cards stand for churn.csv's 1,500 rows, 50 rows a card: 24 cards
// (1,200 rows) train, 6 (300 rows) are locked away until the winning model is scored on them, once.
//
//   node scripts/heldout/build.mjs      writes components/site/heldout-art.ts
//
// iso-kit.mjs is the Anatomy kit's core, copied in; the drawing is static, and HeldOut.tsx moves the parts.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./iso-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "../../components/site/heldout-art.ts");

const WIDTH = 640;
const HEIGHT = 440;

// --- the model: rows, cards, the split ---------------------------------------------------------------
const ROWS = 1500;
const CARDS = 30;
const PER_CARD = ROWS / CARDS; // 50 rows a card
const EVERY = 5; // every fifth card is a test card: 300 of 1,500
const TEST = Array.from({ length: CARDS }, (_, i) => (i + 1) % EVERY === 0);
const TRAIN_N = TEST.filter((t) => !t).length; // 24
const TEST_N = CARDS - TRAIN_N; // 6

// --- the world (x along the bench, y towards the front, z up; 1 unit ~ 1 px) --------------------------
const PLATE = { x: -176, y: -78, w: 352, d: 168, r: 9 };
const PLATE_H = 12;
const FOOT_H = 5;

const CARD = { w: 34, d: 38 };
const CARD_T = 1.4;

const CH_Z = 30; // the channel floor, on four posts
const CH = { x: -118, y: -24, w: 168, d: 48, r: 2 };
const CH_FLOOR = 3;
const WALL_H = 7;
const GAP = [-34, 6]; // the opening in the channel's front wall, where test cards leave

const HOP = { x: -164, y: -24, w: 46, d: 48, r: 2 };
const HOP_TOP = CH_Z + 74;
const HOP_STACK = 12; // slabs in the hopper at the start

const CAB = { x: -54, y: 36, w: 66, d: 40, r: 3 }; // the locked drawer's cabinet
const CAB_H = 24;
const FRONT_Y = CAB.y + CAB.d;

const TRAY = { x: 66, y: -34, w: 94, d: 68, r: 4 };
const TRAY_H = 14;

const CARD_AT = { x: -158, y: -19 }; // a card's plan corner when it sits in the hopper
const TRAY_CARD = { x: TRAY.x + (TRAY.w - CARD.w) / 2, y: TRAY.y + (TRAY.d - CARD.d) / 2 };

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [HOP.x, HOP.y, HOP_TOP + 10], [HOP.x + HOP.w, HOP.y + HOP.d, HOP_TOP + 10], [TRAY.x + TRAY.w, 0, 40]],
  WIDTH,
  HEIGHT,
  { pad: 22, azimuth: 40 },
);

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const slab = (plan, z, h, steps = 3, bevel = 0.5) => k.slabOf(plan, z, h, P, steps, bevel);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

// --- the bench plate ----------------------------------------------------------------------------------
function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 24))) {
    out.push(S(k.cylinder(x, y, 7.5, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.6, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 11);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.3, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // a ruler strip along the front edge, ticks every 5, a major every 25
  const strip = { x: PLATE.x + 30, y: PLATE.y + PLATE.d - 15, w: PLATE.w - 60, d: 7, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 5, 5, strip.y + 0.8, 0.6, [2, 3.8], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(D([at([strip.x + 2, strip.y + 3.5, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.5, 0.6])], { size: 0.45 }));
  // a label plate at the back right, three rule lines standing for its engraving
  const label = { x: 98, y: PLATE.y + 12, w: 52, d: 18, r: 1.5 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 3, 0.2), { tone: "lo", crease: "none" }));
  out.push(L([0, 1, 2].map((i) => k.lineOnTop([label.x + 5, label.y + 5 + i * 4], [label.x + (i === 2 ? 30 : 46), label.y + 5 + i * 4], 0.8, P)).join(""), { tone: "faint", free: "both" }));
  out.push(D(k.corners(label, 2.4).map(([x, y]) => at([x, y, 0.8])), { size: 0.35 }));
  return out.join("");
}

// --- the training tray (behind the channel's end) -------------------------------------------------------
function tray() {
  const out = [`<path class="iso-halo" d="${k.haloOf(k.insetPlan(TRAY, 2), 0, 0.1, P)}" style="filter:blur(4px)"/>`];
  out.push(S(slab(TRAY, 0, 2, 6, 0.6), { tone: "mid" }));
  // four walls, drawn back to front
  const t = 3;
  out.push(S(slab({ x: TRAY.x, y: TRAY.y, w: TRAY.w, d: t, r: 0.6 }, 2, TRAY_H - 2), { tone: "mid" }));
  out.push(S(slab({ x: TRAY.x, y: TRAY.y, w: t, d: TRAY.d, r: 0.6 }, 2, TRAY_H - 2), { tone: "mid" }));
  return out.join("");
}

function trayFront() {
  const out = [];
  const t = 3;
  out.push(S(slab({ x: TRAY.x + TRAY.w - t, y: TRAY.y, w: t, d: TRAY.d, r: 0.6 }, 2, TRAY_H - 2), { tone: "mid" }));
  const front = { x: TRAY.x, y: TRAY.y + TRAY.d - t, w: TRAY.w, d: t, r: 0.6 };
  out.push(S(slab(front, 2, TRAY_H - 2), { tone: "mid" }));
  // a finger notch and two rivets on the front wall
  out.push(D([at([TRAY.x + 10, TRAY.y + TRAY.d, 9]), at([TRAY.x + TRAY.w - 10, TRAY.y + TRAY.d, 9])], { size: 0.45 }));
  out.push(L(k.sideRing([TRAY.x + TRAY.w / 2, TRAY.y + TRAY.d + 0.1, 9], 3.2, "y", P, 20), { tone: "lo" }));
  return out.join("");
}

// the training stack: one slab per training card, shown as cards arrive
function trayStack() {
  const out = [];
  for (let i = 0; i < TRAIN_N; i++) {
    const z = 2 + i * (CARD_T + 0.35);
    const jitter = ((i * 37) % 5) - 2;
    const plan = { x: TRAY_CARD.x + jitter * 0.6, y: TRAY_CARD.y - jitter * 0.4, w: CARD.w, d: CARD.d, r: 1 };
    out.push(`<g data-tray="${i}" style="opacity:0">${S(k.slabOf(plan, z, CARD_T, P, 3, 0.2), { tone: "hi", crease: "lo" })}</g>`);
  }
  return out.join("");
}

// --- the channel on its posts ---------------------------------------------------------------------------
function channel() {
  const out = [];
  const posts = byDepth([[CH.x + 8, CH.y + 6], [CH.x + CH.w - 8, CH.y + 6], [CH.x + 8, CH.y + CH.d - 6], [CH.x + CH.w - 8, CH.y + CH.d - 6]]);
  for (const [x, y] of posts) {
    out.push(S(k.cylinder(x, y, 4.2, 0, 1.4, P, 24), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 2.6, 1.4, CH_Z - 1.4, P, 20), { tone: "mid", crease: "none" }));
    out.push(L(k.sideArc(x, y, 2.6, CH_Z - 8, P, 12), { tone: "faint" }));
  }
  out.push(S(k.slabOf(CH, CH_Z, CH_FLOOR, P, 4, 0.6), { tone: "mid" }));
  // two guide grooves along the floor
  out.push(L([CH.y + 12, CH.y + CH.d - 12].map((y) => k.lineOnTop([CH.x + 4, y], [CH.x + CH.w - 4, y], CH_Z + CH_FLOOR, P)).join(""), { tone: "faint" }));
  // the back wall, whole
  out.push(S(slab({ x: CH.x, y: CH.y - 3, w: CH.w, d: 3, r: 0.6 }, CH_Z, CH_FLOOR + WALL_H), { tone: "mid" }));
  out.push(D([CH.x + 20, CH.x + CH.w / 2, CH.x + CH.w - 20].map((x) => at([x, CH.y - 1.5, CH_Z + CH_FLOOR + WALL_H])), { size: 0.4 }));
  return out.join("");
}

function channelFront() {
  const out = [];
  const y = CH.y + CH.d;
  // the front wall, open between GAP[0] and GAP[1] for the test cards
  out.push(S(slab({ x: CH.x, y, w: GAP[0] - CH.x, d: 3, r: 0.6 }, CH_Z, CH_FLOOR + WALL_H), { tone: "mid" }));
  out.push(S(slab({ x: GAP[1], y, w: CH.x + CH.w - GAP[1], d: 3, r: 0.6 }, CH_Z, CH_FLOOR + WALL_H), { tone: "mid" }));
  out.push(D([CH.x + 20, GAP[1] + 20].map((x) => at([x, y + 1.5, CH_Z + CH_FLOOR + WALL_H])), { size: 0.4 }));
  // the chute from the opening down to the cabinet's slot
  const chute = { x: GAP[0] + 2, y: y + 3, w: GAP[1] - GAP[0] - 4, d: CAB.y + 8 - (y + 3), r: 0.5 };
  out.push(S(slab(chute, CAB_H + 2, 1.4, 3, 0.2), { tone: "lo" }));
  return out.join("");
}

// --- the hopper and its stack ---------------------------------------------------------------------------
function hopper() {
  const out = [];
  out.push(S(slab({ x: HOP.x - 2, y: HOP.y - 2, w: HOP.w + 4, d: HOP.d + 4, r: 2 }, CH_Z, CH_FLOOR, 4, 0.4), { tone: "mid" }));
  // the hopper's back posts
  for (const [x, y] of [[HOP.x, HOP.y], [HOP.x + HOP.w, HOP.y]]) {
    out.push(S(slab({ x: x - 2, y: y - 2, w: 4, d: 4, r: 0.8 }, CH_Z + CH_FLOOR, HOP_TOP - CH_Z - CH_FLOOR, 3, 0.3), { tone: "mid" }));
  }
  out.push(S(slab({ x: HOP.x - 2, y: HOP.y - 2, w: 4, d: HOP.d + 4, r: 0.8 }, CH_Z + CH_FLOOR, HOP_TOP - CH_Z - CH_FLOOR, 3, 0.3), { tone: "mid" }));
  return out.join("");
}

function hopperStack() {
  const out = [];
  for (let i = 0; i < HOP_STACK; i++) {
    const z = CH_Z + CH_FLOOR + i * (CARD_T + 0.9);
    out.push(`<g data-hop="${i}">${S(k.slabOf({ x: CARD_AT.x, y: CARD_AT.y, w: CARD.w, d: CARD.d, r: 1 }, z, CARD_T, P, 3, 0.2), { tone: "hi", crease: "lo" })}</g>`);
  }
  return out.join("");
}

function hopperFront() {
  const out = [];
  for (const [x, y] of [[HOP.x, HOP.y + HOP.d], [HOP.x + HOP.w, HOP.y + HOP.d]]) {
    out.push(S(slab({ x: x - 2, y: y - 2, w: 4, d: 4, r: 0.8 }, CH_Z + CH_FLOOR + 8, HOP_TOP - CH_Z - CH_FLOOR - 8, 3, 0.3), { tone: "mid" }));
  }
  // the top frame, and a knurled knob on its corner
  const top = { x: HOP.x - 3, y: HOP.y - 3, w: HOP.w + 6, d: HOP.d + 6, r: 2 };
  out.push(S(slab(top, HOP_TOP, 3, 4, 0.4), { tone: "mid" }));
  out.push(S(slab(k.insetPlan(top, 6), HOP_TOP, 3.2, 4, 0.2), { tone: "lo", crease: "none" }));
  const knob = [HOP.x + HOP.w - 4, HOP.y + HOP.d - 4];
  out.push(S(k.cylinder(knob[0], knob[1], 4, HOP_TOP + 3, 4, P, 28), { tone: "mid" }));
  out.push(L(k.knurl(knob[0], knob[1], 4, HOP_TOP + 3, HOP_TOP + 7, 28, P), { tone: "lo" }));
  out.push(D(k.corners(top, 2.2).map(([x, y]) => at([x, y, HOP_TOP + 3])), { size: 0.4 }));
  return out.join("");
}

// --- the pusher behind the hopper -----------------------------------------------------------------------
const PUSH_FROM = HOP.x - 16;
function pusherRail() {
  const out = [];
  const rail = { x: PLATE.x + 6, y: -4, w: HOP.x - PLATE.x - 4, d: 8, r: 1 };
  out.push(S(slab(rail, CH_Z, 2, 3, 0.3), { tone: "lo" }));
  out.push(S(k.cylinder(rail.x + 6, 0, 3, 0, CH_Z, P, 18), { tone: "mid", crease: "none" }));
  return out.join("");
}
function pusher() {
  const out = [];
  const body = { x: -10, y: -14, w: 10, d: 28, r: 1.5 };
  out.push(S(slab(body, CH_Z + 2, 7, 3, 0.4), { tone: "mid" }));
  out.push(S(slab({ x: -14, y: -3, w: 4, d: 6, r: 1 }, CH_Z + 4, 3, 3, 0.2), { tone: "mid" }));
  out.push(S(k.cylinder(-16, 0, 2.6, CH_Z + 3, 5, P, 20), { tone: "mid" }));
  out.push(L(k.knurl(-16, 0, 2.6, CH_Z + 3, CH_Z + 8, 16, P), { tone: "lo" }));
  out.push(D([at([-5, -9, CH_Z + 9]), at([-5, 9, CH_Z + 9])], { size: 0.4 }));
  return out.join("");
}

// --- the flap: two poses, crossfaded ------------------------------------------------------------------
const HINGE = [GAP[1] + 6, CH.y + 3];
function flap(pose) {
  // pass: folded flat against the back wall; turn: swung across the channel to the front opening
  const end = pose === "pass" ? [HINGE[0] - 34, CH.y + 3] : [GAP[0] + 10, CH.y + CH.d - 4];
  const dx = end[0] - HINGE[0], dy = end[1] - HINGE[1];
  const len = Math.hypot(dx, dy), nx = (-dy / len) * 1.2, ny = (dx / len) * 1.2;
  const ring = [[HINGE[0] + nx, HINGE[1] + ny], [end[0] + nx, end[1] + ny], [end[0] - nx, end[1] - ny], [HINGE[0] - nx, HINGE[1] - ny]];
  const solid = k.extrude(ring, CH_Z + CH_FLOOR, WALL_H - 1, P, {});
  return S(solid, { tone: "hi", crease: "lo" });
}
function hinge() {
  const out = [];
  out.push(S(k.cylinder(HINGE[0], HINGE[1], 2.4, CH_Z + CH_FLOOR, WALL_H + 1, P, 20), { tone: "mid" }));
  out.push(D([at([HINGE[0], HINGE[1], CH_Z + CH_FLOOR + WALL_H + 1])], { size: 0.5, tone: "hi" }));
  return out.join("");
}

// --- the locked cabinet and its drawer -------------------------------------------------------------------
function cabinet() {
  const out = [`<path class="iso-halo" d="${k.haloOf(k.insetPlan(CAB, 2), 0, 0.1, P)}" style="filter:blur(4px)"/>`];
  out.push(S(k.slabOf(CAB, 0, CAB_H, P, 5, 1), { tone: "mid" }));
  // the slot in the top that the test cards drop through
  const slot = { x: GAP[0] + 2, y: CAB.y + 5, w: GAP[1] - GAP[0] - 4, d: 4, r: 1.2 };
  out.push(S(k.slabOf(slot, CAB_H, 0.2, P, 3), { tone: "lo", crease: "none", flat: true }));
  out.push(D(k.corners(CAB, 4).map(([x, y]) => at([x, y, CAB_H])), { size: 0.4 }));
  // the opening the drawer runs in, a shade darker than the face
  out.push(S(k.slabOf({ x: CAB.x + 3.4, y: FRONT_Y - 0.2, w: CAB.w - 6.8, d: 0.2, r: 0.4 }, 2.4, CAB_H - 4.8, P, 3), { tone: "lo", crease: "none", flat: true }));
  return out.join("");
}

// the drawer: an open box holding the test cards, behind a front with its pull and lock
const DRAWER_OUT = 30; // how far it slides out when it is read
function drawerBox() {
  const out = [];
  const box = { x: CAB.x + 6, y: CAB.y + 4, w: CAB.w - 12, d: FRONT_Y - CAB.y - 4, r: 1 };
  out.push(S(slab(box, 3, 1.4, 3, 0.2), { tone: "mid" }));
  out.push(S(slab({ x: box.x, y: box.y, w: box.w, d: 2, r: 0.4 }, 4.4, CAB_H - 12, 3, 0.2), { tone: "mid" }));
  out.push(S(slab({ x: box.x, y: box.y, w: 2, d: box.d, r: 0.4 }, 4.4, CAB_H - 12, 3, 0.2), { tone: "mid" }));
  out.push(S(slab({ x: box.x + box.w - 2, y: box.y, w: 2, d: box.d, r: 0.4 }, 4.4, CAB_H - 12, 3, 0.2), { tone: "mid" }));
  // the six test cards, gold: what no model trains on
  for (let i = 0; i < TEST_N; i++) {
    const plan = { x: box.x + (box.w - CARD.w) / 2 + ((i * 3) % 3) - 1, y: box.y + 2 + ((i * 7) % 3) * 0.4, w: CARD.w, d: Math.min(CARD.d, box.d - 4), r: 1 };
    out.push(S(k.slabOf(plan, 4.6 + i * (CARD_T + 0.3), CARD_T, P, 3, 0.2), { tone: "hi", lit: true }));
  }
  return out.join("");
}
function drawerFront() {
  const out = [];
  const front = { x: CAB.x + 4, y: FRONT_Y, w: CAB.w - 8, d: 2.2, r: 0.6 };
  out.push(S(slab(front, 3, CAB_H - 6, 3, 0.3), { tone: "hi", crease: "lo" }));
  out.push(S(slab({ x: CAB.x + 12, y: FRONT_Y + 2.2, w: 18, d: 2.4, r: 0.8 }, 9, 3, 3, 0.3), { tone: "mid" }));
  const lock = [CAB.x + CAB.w - 16, FRONT_Y + 2.25, 13];
  out.push(L(k.sideRing(lock, 4, "y", P, 28), { tone: "mid" }));
  out.push(L(k.sideRing(lock, 2.6, "y", P, 24), { tone: "lo" }));
  out.push(D([at(lock)], { size: 0.7, tone: "hi" }));
  out.push(D([at([front.x + 3, FRONT_Y + 2.2, 6]), at([front.x + front.w - 3, FRONT_Y + 2.2, 6]), at([front.x + 3, FRONT_Y + 2.2, CAB_H - 6]), at([front.x + front.w - 3, FRONT_Y + 2.2, CAB_H - 6])], { size: 0.35 }));
  return out.join("");
}
const LOCK = [CAB.x + CAB.w - 16, FRONT_Y + 2.4, 13];
function latch(pose) {
  // locked: the bolt lies across; open: turned upright
  const [x, y, z] = LOCK;
  const plan = pose === "locked" ? { x: x - 1, y, w: 11, d: 1.4, r: 0.5 } : { x: x - 1, y, w: 2, d: 1.4, r: 0.5 };
  const h = pose === "locked" ? 2 : 10;
  const z0 = pose === "locked" ? z - 1 : z - 1;
  return S(k.slabOf(plan, z0, h, P, 3, 0.2), { tone: "hi", crease: "lo" });
}

// --- the moving card ----------------------------------------------------------------------------------
function card() {
  return S(k.slabOf({ x: 0, y: 0, w: CARD.w, d: CARD.d, r: 1 }, 0, CARD_T, P, 3, 0.2), { tone: "hi", lit: true });
}

// --- assemble, back to front ---------------------------------------------------------------------------
const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "A card sorter on a bench. Row cards drop from a hopper into a raised channel; a pusher sends them along it. A hinged flap turns every fifth card out through the front of the channel and down a chute into a locked drawer; the rest slide on into an open training tray.",
  body: [
    plate(),
    tray(),
    trayStack(),
    pusherRail(),
    `<g data-part="pusher" transform="${k.translateAlong(P, [PUSH_FROM, 0, 0])}">${pusher()}</g>`,
    channel(),
    hopper(),
    hopperStack(),
    `<g data-part="card" style="opacity:0">${card()}</g>`,
    `<g data-part="flap-pass">${flap("pass")}</g>`,
    `<g data-part="flap-turn" style="opacity:0">${flap("turn")}</g>`,
    hinge(),
    hopperFront(),
    channelFront(),
    trayFront(),
    cabinet(),
    `<g data-part="drawer"><g data-part="drawer-box" style="opacity:0">${drawerBox()}</g>${drawerFront()}<g data-part="latch-locked">${latch("locked")}</g><g data-part="latch-open" style="opacity:0">${latch("open")}</g></g>`,
  ],
});

// the card's route, in world points (its plan corner and floor height)
const exitX = CH.x + 8;
const flapX = GAP[0] + 8;
const ROUTES = {
  start: [CARD_AT.x, CARD_AT.y, CH_Z + CH_FLOOR],
  out: [exitX, CARD_AT.y, CH_Z + CH_FLOOR],
  flap: [flapX - CARD.w / 2, CARD_AT.y, CH_Z + CH_FLOOR],
  lip: [flapX - CARD.w / 2, CH.y + CH.d - 6, CH_Z + CH_FLOOR],
  slot: [flapX - CARD.w / 2, CAB.y + 4, CAB_H + 3],
  sunk: [flapX - CARD.w / 2, CAB.y + 4, CAB_H - 10],
  end: [CH.x + CH.w - CARD.w + 6, CARD_AT.y, CH_Z + CH_FLOOR],
  tray: [TRAY_CARD.x, TRAY_CARD.y, 2],
};
const offsets = Object.fromEntries(Object.entries(ROUTES).map(([name, p]) => [name, k.axisVector(P, p)]));
const lift = (n) => k.axisVector(P, [0, 0, n]);
const trayLift = Array.from({ length: TRAIN_N }, (_, i) => k.axisVector(P, [0, 0, i * (CARD_T + 0.35)]));
const pushAt = (x) => k.axisVector(P, [x, 0, 0]);

const art = {
  width: WIDTH,
  height: HEIGHT,
  rows: ROWS,
  cards: CARDS,
  perCard: PER_CARD,
  test: TEST,
  hopStack: HOP_STACK,
  offsets,
  trayLift,
  push: { from: pushAt(PUSH_FROM), to: pushAt(CARD_AT.x - 2) },
  cardLift: lift(1),
  drawerOut: k.axisVector(P, [0, DRAWER_OUT, 0]),
  svg,
};

writeFileSync(
  OUT,
  `// Generated by scripts/heldout/build.mjs (the Anatomy kit); do not edit by hand.\n/* eslint-disable */\nexport const ART = ${JSON.stringify(art)} as const;\n`,
);
console.log(`wrote ${OUT} (${(svg.length / 1024).toFixed(1)} KB of SVG; ${TRAIN_N} train cards, ${TEST_N} test cards)`);
