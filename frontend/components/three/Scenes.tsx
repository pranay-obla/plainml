"use client";

// The 3D views of a run's results: bars (models, columns, drift), points (groups, unusual rows)
// and series (forecasts, one lane per series). Loaded only when a run page shows one.

import { Html, Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import * as THREE from "three";
import type { Bar, Visual } from "@/lib/types";
import { num } from "@/lib/format";
import { seriesColor, usePalette, useReducedMotionPref, type Palette, type Tone } from "./theme";

type Bars = Extract<Visual, { type: "bars" }>;
type Points = Extract<Visual, { type: "points" }>;
type Series = Extract<Visual, { type: "series" }>;

function Tip({ position, title, detail }: { position: [number, number, number]; title: string; detail?: string }) {
  return (
    <Html position={position} zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
      <div className="tip">
        <b>{title}</b>
        {detail}
      </div>
    </Html>
  );
}

// --- fixed labels: drawn as plain HTML on top of the canvas, moved each frame -------------------------

interface Label {
  text: string;
  at: [number, number, number];
  style?: "below" | "right" | "left";
}

function Projector({ labels, refs }: { labels: Label[]; refs: MutableRefObject<(HTMLDivElement | null)[]> }) {
  const { camera, size } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    labels.forEach((label, i) => {
      const el = refs.current[i];
      if (!el) return;
      v.set(...label.at).project(camera);
      const hidden = v.z > 1;
      el.style.visibility = hidden ? "hidden" : "visible";
      el.style.left = `${((v.x + 1) / 2) * size.width}px`;
      el.style.top = `${((1 - v.y) / 2) * size.height}px`;
    });
  });
  return null;
}

function LabelLayer({ labels, refs }: { labels: Label[]; refs: MutableRefObject<(HTMLDivElement | null)[]> }) {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden" }} aria-hidden="true">
      {labels.map((label, i) => (
        <div
          key={`${label.text}-${i}`}
          ref={(el) => {
            refs.current[i] = el;
          }}
          className={label.style === "left" ? "axis-label left" : "axis-label"}
          style={{
            position: "absolute",
            visibility: "hidden",
            transform:
              label.style === "below"
                ? "translate(-100%, 4px) rotate(-38deg)"
                : label.style === "right"
                  ? "translate(8px, -50%)"
                  : undefined,
            transformOrigin: label.style === "below" ? "100% 0" : undefined,
          }}
        >
          {label.text}
        </div>
      ))}
    </div>
  );
}

// --- bars ------------------------------------------------------------------------------------------

const HEIGHT = 3.2;

function barColor(p: Palette, bar: Bar) {
  if (bar.tone === "best" || bar.tone === "picked") return p.highlight;
  if (bar.tone === "warn") return p.warn;
  if (bar.tone === "bad") return p.bad;
  return p.other;
}

function shortLabel(text: string, max = 16) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function barLayout(visual: Bars) {
  const top = Math.max(...visual.bars.map((b) => Math.abs(b.value)), ...(visual.lines || []).map((l) => Math.abs(l.value)), 1e-9);
  const gap = 0.75;
  const width = visual.bars.length * gap;
  return {
    gap,
    width,
    origin: [-width / 2 + gap / 2, -1.6, 0] as [number, number, number],
    scale: (v: number) => Math.max(0.03, (Math.max(v, 0) / top) * HEIGHT),
  };
}

function barLabels(visual: Bars): Label[] {
  const { gap, width, origin, scale } = barLayout(visual);
  const every = visual.bars.length > 14 ? 2 : 1;
  const labels: Label[] = visual.bars
    .map((bar, i) => ({ bar, i }))
    .filter(({ bar, i }) => i % every === 0 || bar.tone === "best" || bar.tone === "picked")
    .map(({ bar, i }) => ({ text: shortLabel(bar.label), at: [origin[0] + i * gap, origin[1] - 0.05, 0.3], style: "below" }));
  for (const line of visual.lines || []) {
    labels.push({ text: `${line.label}: ${num(line.value)}`, at: [origin[0] + width - gap / 2, origin[1] + scale(line.value), 0], style: "right" });
  }
  return labels;
}

function BarsView({ visual, palette }: { visual: Bars; palette: Palette }) {
  const [hover, setHover] = useState<number | null>(null);
  const bars = visual.bars;
  const { gap, width, origin, scale } = barLayout(visual);
  return (
    <group position={origin}>
      {bars.map((bar, i) => {
        const h = scale(bar.value);
        return (
          <group key={bar.label} position={[i * gap, 0, 0]}>
            <mesh
              position={[0, h / 2, 0]}
              onPointerOver={(e: ThreeEvent<PointerEvent>) => {
                e.stopPropagation();
                setHover(i);
              }}
              onPointerOut={() => setHover((v) => (v === i ? null : v))}
            >
              <boxGeometry args={[0.5, h, 0.5]} />
              <meshStandardMaterial color={barColor(palette, bar)} roughness={0.55} metalness={0.05} emissive={hover === i ? barColor(palette, bar) : "#000000"} emissiveIntensity={hover === i ? 0.35 : 0} />
            </mesh>
            {hover === i ? <Tip position={[0, h + 0.2, 0]} title={bar.label} detail={`${visual.value_label}: ${num(bar.value)}${bar.note ? ` · ${bar.note}` : ""}`} /> : null}
          </group>
        );
      })}
      {(visual.lines || []).map((line) => {
        const y = scale(line.value);
        return (
          <Line key={line.label} points={[[-gap, y, 0], [width - gap / 2, y, 0]]} color={palette.ink2} lineWidth={1.5} dashed dashSize={0.18} gapSize={0.12} />
        );
      })}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[width / 2 - gap / 2, -0.001, 0]}>
        <planeGeometry args={[width + gap, 2.2]} />
        <meshBasicMaterial color={palette.line} transparent opacity={palette.dark ? 0.5 : 0.7} depthWrite={false} />
      </mesh>
    </group>
  );
}

// --- points ----------------------------------------------------------------------------------------

function normalise(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] ?? 0;
  const lo = q(0.02);
  const hi = q(0.98);
  const mid = (lo + hi) / 2;
  const half = Math.max((hi - lo) / 2, 1e-9);
  return values.map((v) => Math.max(-1.6, Math.min(1.6, (v - mid) / half)) * 2.2);
}

function PointsView({ visual, palette }: { visual: Points; palette: Palette }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const [hover, setHover] = useState<number | null>(null);
  const flagged = Boolean(visual.size);
  const positions = useMemo(() => {
    const x = normalise(visual.x);
    const y = normalise(visual.y);
    const z = visual.z.some((v) => v !== 0) ? normalise(visual.z) : visual.z.map(() => 0);
    return x.map((_, i) => [x[i], y[i], z[i]] as [number, number, number]);
  }, [visual]);

  useLayoutEffect(() => {
    if (!mesh.current) return;
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    positions.forEach((p, i) => {
      const big = flagged && visual.group[i] === 1;
      const s = big ? 1.9 : 1;
      matrix.makeScale(s, s, s).setPosition(...p);
      mesh.current!.setMatrixAt(i, matrix);
      color.set(flagged ? (big ? palette.highlight : palette.other) : seriesColor(palette, visual.group[i]));
      mesh.current!.setColorAt(i, color);
    });
    mesh.current.instanceMatrix.needsUpdate = true;
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true;
  }, [positions, palette, visual, flagged]);

  return (
    <group>
      <instancedMesh
        ref={mesh}
        args={[undefined, undefined, positions.length]}
        onPointerMove={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation();
          if (e.instanceId !== undefined) setHover(e.instanceId);
        }}
        onPointerOut={() => setHover(null)}
      >
        <sphereGeometry args={[0.055, 10, 8]} />
        <meshStandardMaterial roughness={0.45} metalness={0.05} transparent opacity={flagged ? 0.9 : 0.92} />
      </instancedMesh>
      {hover !== null ? <Tip position={positions[hover]} title={visual.groups[visual.group[hover]]} detail={visual.hover?.[hover]} /> : null}
      <gridHelper args={[7, 14, palette.line, palette.line]} position={[0, -2.6, 0]} />
    </group>
  );
}

// --- series ----------------------------------------------------------------------------------------

function seriesLayout(visual: Series) {
  const lanes = visual.lanes.slice(0, 6);
  const all = lanes.flatMap((l) => [...l.history.y, ...l.forecast.y, ...(l.forecast.lower || []), ...(l.forecast.upper || [])]).filter(Number.isFinite);
  const lo = Math.min(...all, 0);
  const hi = Math.max(...all, 1e-9);
  const times = lanes.flatMap((l) => [...l.history.x, ...l.forecast.x]);
  const t0 = Math.min(...times);
  const t1 = Math.max(...times, t0 + 1);
  const X = (t: number) => ((t - t0) / (t1 - t0)) * 8 - 4;
  const Y = (v: number) => ((v - lo) / (hi - lo)) * 3 - 1.6;
  const depth = 1.3;
  const laneZ = (li: number) => (li - (lanes.length - 1) / 2) * depth;
  return { lanes, X, Y, lo, depth, laneZ };
}

function seriesLabels(visual: Series): Label[] {
  const { lanes, X, Y, lo, laneZ } = seriesLayout(visual);
  return lanes.map((lane, li) => ({
    text: shortLabel(lane.name, 18),
    at: [X(lane.history.x[0] ?? 0) - 0.15, Y(lane.history.y[0] ?? lo), laneZ(li)],
    style: "left",
  }));
}

function SeriesView({ visual, palette }: { visual: Series; palette: Palette }) {
  const [hover, setHover] = useState<string | null>(null);
  const { lanes, X, Y, depth, laneZ } = seriesLayout(visual);
  return (
    <group>
      {lanes.map((lane, li) => {
        const z = laneZ(li);
        const history = lane.history.x.map((t, i) => [X(t), Y(lane.history.y[i]), z] as [number, number, number]);
        const lastHistory = history[history.length - 1];
        const forecast = [...(lastHistory ? [lastHistory] : []), ...lane.forecast.x.map((t, i) => [X(t), Y(lane.forecast.y[i]), z] as [number, number, number])];
        let band: THREE.BufferGeometry | null = null;
        if (lane.forecast.lower && lane.forecast.upper && lane.forecast.x.length > 1) {
          const vertices: number[] = [];
          for (let i = 0; i < lane.forecast.x.length - 1; i++) {
            const a = X(lane.forecast.x[i]);
            const b = X(lane.forecast.x[i + 1]);
            const al = Y(lane.forecast.lower[i]);
            const au = Y(lane.forecast.upper[i]);
            const bl = Y(lane.forecast.lower[i + 1]);
            const bu = Y(lane.forecast.upper[i + 1]);
            vertices.push(a, al, z, b, bl, z, b, bu, z, a, al, z, b, bu, z, a, au, z);
          }
          band = new THREE.BufferGeometry();
          band.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
        }
        const on = hover === lane.name;
        const end = lane.forecast.y.length - 1;
        return (
          <group key={lane.name} onPointerOver={(e) => (e.stopPropagation(), setHover(lane.name))} onPointerOut={() => setHover(null)}>
            {history.length > 1 ? <Line points={history} color={palette.ink2} lineWidth={on ? 2.5 : 1.6} /> : null}
            {forecast.length > 1 ? <Line points={forecast} color={palette.highlight} lineWidth={on ? 3.2 : 2.2} /> : null}
            {band ? (
              <mesh geometry={band}>
                <meshBasicMaterial color={palette.highlight} transparent opacity={on ? 0.32 : 0.2} side={THREE.DoubleSide} depthWrite={false} />
              </mesh>
            ) : null}
            {on && end >= 0 ? (
              <Tip
                position={forecast[forecast.length - 1]}
                title={lane.name}
                detail={`${visual.value_label} at the end: ${num(lane.forecast.y[end])}${lane.forecast.lower && lane.forecast.upper ? ` (${num(lane.forecast.lower[end])} – ${num(lane.forecast.upper[end])})` : ""}`}
              />
            ) : null}
          </group>
        );
      })}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.62, 0]}>
        <planeGeometry args={[9, lanes.length * depth + 0.6]} />
        <meshBasicMaterial color={palette.line} transparent opacity={palette.dark ? 0.4 : 0.6} depthWrite={false} />
      </mesh>
    </group>
  );
}

export default function Scenes({ visual, tone, spin, far = 1 }: { visual: Visual; tone?: Tone; spin?: boolean; far?: number }) {
  const palette = usePalette(tone);
  const still = useReducedMotionPref();
  const labels = useMemo(() => (visual.type === "bars" ? barLabels(visual) : visual.type === "series" ? seriesLabels(visual) : []), [visual]);
  const labelRefs = useRef<(HTMLDivElement | null)[]>([]);
  if (!palette) return null;
  const base: [number, number, number] = visual.type === "series" ? [0, 2.6, 7.5] : visual.type === "points" ? [4.8, 3.2, 6.4] : [0, 2.2, 8.2];
  const camera = base.map((v) => v * far) as [number, number, number]; // `far` > 1 backs the camera off (a narrow stage)
  return (
    <div style={{ position: "absolute", inset: 0 }}>
    <Canvas camera={{ position: camera, fov: 42 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }}>
      <Projector labels={labels} refs={labelRefs} />
      <ambientLight intensity={palette.dark ? 0.7 : 0.95} />
      <directionalLight position={[4, 8, 6]} intensity={palette.dark ? 1.4 : 1.1} />
      <directionalLight position={[-6, 3, -4]} intensity={0.35} />
      {visual.type === "bars" ? <BarsView visual={visual} palette={palette} /> : null}
      {visual.type === "points" ? <PointsView visual={visual} palette={palette} /> : null}
      {visual.type === "series" ? <SeriesView visual={visual} palette={palette} /> : null}
      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={4}
        maxDistance={16 * far}
        maxPolarAngle={Math.PI / 2.05}
        autoRotate={(spin ?? visual.type === "points") && !still}
        autoRotateSpeed={0.6}
      />
    </Canvas>
    <LabelLayer labels={labels} refs={labelRefs} />
    </div>
  );
}
