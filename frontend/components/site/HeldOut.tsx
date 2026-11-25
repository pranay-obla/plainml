"use client";

// Fig 1 on the home page: the held-out test as a card sorter (scripts/heldout/build.mjs draws it with the
// Anatomy kit). 30 cards stand for churn.csv's 1,500 rows; every fifth goes into the locked drawer.
// The state is a pure function of time, so a frame never depends on the one before it.

import { useEffect, useRef, useState } from "react";
import { ART } from "./heldout-art";
import "./iso.css";

type Flat = readonly [number, number];
const O = ART.offsets as unknown as Record<string, Flat>;
const STEP = 0.62; // seconds a card takes
const SORT = ART.cards * STEP;
const HOLD = 1.2; // after the last card, before the drawer opens
const OPEN = 4.6; // the latch turns, the drawer slides out, is read once, and closes
const RESET = 0.8;
const LOOP = SORT + HOLD + OPEN + RESET;

const ease = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
const mix = (a: Flat, b: Flat, u: number): Flat => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
const clamp01 = (u: number) => Math.min(1, Math.max(0, u));

/** Where the moving card is at fraction u of its step, along a route of named points. */
function along(route: Flat[], u: number): Flat {
  const n = route.length - 1;
  const x = clamp01(u) * n;
  const i = Math.min(n - 1, Math.floor(x));
  return mix(route[i], route[i + 1], ease(x - i));
}

function stateAt(t: number) {
  const time = ((t % LOOP) + LOOP) % LOOP;
  const done = Math.min(ART.cards, Math.floor(time / STEP)); // cards fully sorted
  const u = time < SORT ? (time % STEP) / STEP : 1;
  const index = time < SORT ? done : -1;
  let train = 0;
  let test = 0;
  for (let i = 0; i < done; i++) (ART.test[i] ? test++ : train++);
  const isTest = index >= 0 && ART.test[index];
  const trainSlot = train; // the slab this card lands on
  const openT = time - SORT - HOLD;
  const opening = clamp01(openT / 0.4) * (1 - clamp01((openT - (OPEN - 0.4)) / 0.4)); // the latch
  const slide = ease(clamp01((openT - 0.45) / 0.7)) * (1 - ease(clamp01((openT - (OPEN - 1.1)) / 0.7))); // the drawer
  const resetting = clamp01((time - SORT - HOLD - OPEN) / RESET);
  return { time, done, u, index, isTest, train, test, trainSlot, opening, slide, resetting };
}

export function HeldOut() {
  const stageRef = useRef<HTMLDivElement>(null);
  const readRef = useRef<HTMLSpanElement>(null);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const svg = stage.querySelector("svg");
    if (!svg) return;
    const part = (name: string) => svg.querySelector<SVGGElement>(`[data-part="${name}"]`);
    const card = part("card");
    const pusher = part("pusher");
    const flapPass = part("flap-pass");
    const flapTurn = part("flap-turn");
    const latchLocked = part("latch-locked");
    const latchOpen = part("latch-open");
    const drawer = part("drawer");
    const drawerBox = part("drawer-box");
    const out = ART.drawerOut as unknown as Flat;
    const hop = Array.from(svg.querySelectorAll<SVGGElement>("[data-hop]"));
    const tray = Array.from(svg.querySelectorAll<SVGGElement>("[data-tray]"));
    const trayLift = ART.trayLift as unknown as Flat[];
    const still = window.matchMedia("(prefers-reduced-motion: reduce)");

    const draw = (t: number) => {
      const s = stateAt(t);
      // the hopper empties as cards leave it, and refills during the reset
      const left = s.resetting > 0 ? ART.hopStack : Math.ceil((ART.hopStack * (ART.cards - s.done - (s.index >= 0 ? 1 : 0))) / ART.cards);
      hop.forEach((g, i) => g.style.setProperty("opacity", i < left ? "1" : "0"));
      // the training tray fills one slab per training card; it clears during the reset
      tray.forEach((g, i) => g.style.setProperty("opacity", String(i < s.train ? 1 - s.resetting : 0)));
      // the flap turns only for a test card
      const turn = s.isTest ? clamp01(s.u / 0.18) * (1 - clamp01((s.u - 0.82) / 0.18)) : 0;
      flapPass?.style.setProperty("opacity", String(1 - turn));
      flapTurn?.style.setProperty("opacity", String(turn));
      // the latch turns while the drawer is read
      latchLocked?.style.setProperty("opacity", String(1 - s.opening));
      latchOpen?.style.setProperty("opacity", String(s.opening));
      // the drawer slides out to show the test cards, then back in
      drawer?.setAttribute("transform", `translate(${(out[0] * s.slide).toFixed(2)} ${(out[1] * s.slide).toFixed(2)})`);
      drawerBox?.style.setProperty("opacity", String(clamp01(s.slide * 3)));
      // the pusher strokes at the start of every step
      if (pusher) {
        const stroke = s.index >= 0 ? Math.sin(Math.PI * clamp01(s.u / 0.3)) : 0;
        const [x, y] = mix(ART.push.from as unknown as Flat, ART.push.to as unknown as Flat, stroke);
        pusher.setAttribute("transform", `translate(${x.toFixed(2)} ${y.toFixed(2)})`);
      }
      // the moving card
      if (card) {
        if (s.index < 0) {
          card.style.setProperty("opacity", "0");
        } else {
          const route = s.isTest ? [O.start, O.out, O.flap, O.lip, O.slot, O.sunk] : [O.start, O.out, O.end, mix(O.tray, O.tray, 0)];
          let [x, y] = along(route as Flat[], s.u);
          if (!s.isTest) {
            const landed = trayLift[Math.min(s.trainSlot, trayLift.length - 1)];
            const k = clamp01((s.u - 0.66) / 0.34);
            x += landed[0] * k;
            y += landed[1] * k;
          }
          card.setAttribute("transform", `translate(${x.toFixed(2)} ${y.toFixed(2)})`);
          const fade = s.isTest ? 1 - clamp01((s.u - 0.85) / 0.15) : 1 - clamp01((s.u - 0.96) / 0.04);
          card.style.setProperty("opacity", String(fade));
        }
      }
      const rows = (n: number) => (n * ART.perCard).toLocaleString("en-US");
      const text =
        s.slide > 0.25
          ? "The winning model meets the 300 locked rows, once."
          : s.opening > 0.2
            ? "Unlocking the drawer"
          : s.resetting > 0
            ? "Next spreadsheet."
            : `${rows(s.train)} rows to train, ${rows(s.test)} locked away`;
      if (readRef.current && readRef.current.textContent !== text) readRef.current.textContent = text;
    };

    // reduced motion: the finished sort, locked, and nothing moves
    const FINAL = SORT + 0.01;
    let frame = 0;
    let clock = 0;
    let last = 0;
    let visible = false;
    const tick = (now: number) => {
      const dt = last ? Math.min((now - last) / 1000, 1 / 20) : 0;
      last = now;
      if (!pausedRef.current) clock += dt;
      draw(clock);
      frame = visible && !still.matches ? requestAnimationFrame(tick) : 0;
    };
    const run = () => {
      if (still.matches) {
        draw(FINAL);
        return;
      }
      if (!frame && visible) {
        last = 0;
        frame = requestAnimationFrame(tick);
      }
    };
    const io = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting;
        if (visible) run();
      },
      { rootMargin: "80px" },
    );
    io.observe(stage);
    draw(still.matches ? FINAL : 0);
    const onMotion = () => run();
    still.addEventListener("change", onMotion);
    return () => {
      io.disconnect();
      still.removeEventListener("change", onMotion);
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);

  const toggle = () => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
  };

  return (
    <figure className="iso fig" data-theme="light">
      <div className="fig__plate">
        <div className="fig__corners" aria-hidden="true">
          <span data-corner="fig">Fig 1</span>
          <span data-corner="title">The held-out drawer</span>
          <span data-corner="hint">{paused ? "Paused. Press to play" : "Press to pause"}</span>
          <span data-corner="readout" ref={readRef}>
            1,200 rows to train, 300 locked away
          </span>
        </div>
        <div
          ref={stageRef}
          className="fig__stage"
          role="button"
          tabIndex={0}
          aria-pressed={paused}
          aria-label="The held-out drawer, animated: press to pause or play"
          onClick={toggle}
          onKeyDown={(e) => {
            if (e.key === " " || e.key === "Enter") {
              e.preventDefault();
              toggle();
            }
          }}
          dangerouslySetInnerHTML={{ __html: ART.svg }}
        />
      </div>
      <figcaption className="fig__caption">
        churn.csv&apos;s 1,500 rows as 30 cards. Every fifth card goes into the locked drawer: 300 rows no model trains on.
      </figcaption>
    </figure>
  );
}
