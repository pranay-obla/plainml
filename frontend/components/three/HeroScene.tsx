"use client";

// The landing page's background: rows of a spreadsheet (a flat grid of points) lift off the page
// and sort themselves into groups, then settle back. "Your data becoming a model."

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { hasWebGL, useOnScreen, usePalette, useReducedMotionPref, type Palette } from "./theme";

const COLS = 44;
const ROWS = 30;
const N = COLS * ROWS;
const CYCLE = 12; // seconds for grid -> groups -> grid

function seeded(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function layouts() {
  const rand = seeded(7);
  const grid = new Float32Array(N * 3);
  const groups = new Float32Array(N * 3);
  const which = new Uint8Array(N);
  const centers = [
    [-2.6, 0.6, -0.4],
    [1.2, 1.9, 0.6],
    [2.4, -1.1, -0.8],
  ];
  for (let i = 0; i < N; i++) {
    const c = i % COLS;
    const r = Math.floor(i / COLS);
    grid[i * 3] = (c - COLS / 2) * 0.24;
    grid[i * 3 + 1] = -1.7;
    grid[i * 3 + 2] = (r - ROWS / 2) * 0.24;
    const g = rand() < 0.45 ? 0 : rand() < 0.55 ? 1 : 2;
    which[i] = g;
    const spread = g === 0 ? 0.85 : 0.65;
    groups[i * 3] = centers[g][0] + gaussian(rand) * spread;
    groups[i * 3 + 1] = centers[g][1] + gaussian(rand) * spread * 0.8;
    groups[i * 3 + 2] = centers[g][2] + gaussian(rand) * spread;
  }
  return { grid, groups, which };
}

function dotTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.55, "rgba(255,255,255,0.95)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function ease(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** 0 = grid, 1 = groups, over one cycle: hold, morph, hold, morph back. Starts grouped. */
function phase(time: number) {
  const t = ((time + CYCLE * 0.45) % CYCLE) / CYCLE;
  if (t < 0.15) return 0;
  if (t < 0.4) return ease((t - 0.15) / 0.25);
  if (t < 0.8) return 1;
  return 1 - ease((t - 0.8) / 0.2);
}

function Cloud({ palette, still }: { palette: Palette; still: boolean }) {
  const { grid, groups, which } = useMemo(layouts, []);
  const texture = useMemo(dotTexture, []);
  const points = useRef<THREE.Points>(null);
  const group = useRef<THREE.Group>(null);
  const { pointer } = useThree();

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(groups), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    return g;
  }, [grid, groups, still]);

  // Colors: muted in the grid, each group's color once sorted.
  const colors = useMemo(() => {
    const muted = new THREE.Color(palette.dark ? "#6e4a42" : "#c7ad94");
    const tones = [new THREE.Color(palette.dark ? "#ff6b5e" : "#990000"), new THREE.Color(palette.gold), new THREE.Color(palette.dark ? "#fff6e3" : "#d43b1e")];
    return { muted, tones };
  }, [palette]);

  const paint = (k: number) => {
    const attribute = geometry.getAttribute("color") as THREE.BufferAttribute;
    const c = new THREE.Color();
    for (let i = 0; i < N; i++) {
      c.copy(colors.muted).lerp(colors.tones[which[i]], k);
      attribute.setXYZ(i, c.r, c.g, c.b);
    }
    attribute.needsUpdate = true;
  };

  useEffect(() => {
    paint(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colors, still]);

  const last = useRef(-1);
  useFrame((state, delta) => {
    if (!group.current) return;
    if (still) {
      group.current.rotation.y = 0.35;
      return;
    }
    const k = phase(state.clock.elapsedTime);
    if (Math.abs(k - last.current) > 0.001) {
      const position = geometry.getAttribute("position") as THREE.BufferAttribute;
      const array = position.array as Float32Array;
      for (let i = 0; i < N * 3; i++) {
        // stagger by row, so the spreadsheet lifts off one row after another
        const row = Math.floor(i / 3 / COLS) / ROWS;
        const local = Math.min(1, Math.max(0, k * 1.6 - row * 0.6));
        array[i] = grid[i] + (groups[i] - grid[i]) * ease(local);
      }
      position.needsUpdate = true;
      paint(k);
      last.current = k;
    }
    group.current.rotation.y += delta * 0.06;
    group.current.rotation.x += (pointer.y * 0.12 - group.current.rotation.x) * 0.04;
    group.current.position.x += (2.6 + pointer.x * 0.25 - group.current.position.x) * 0.04;
  });

  return (
    <group ref={group} position={[2.6, 0.2, 0]}>
      <points ref={points} geometry={geometry}>
        <pointsMaterial
          size={palette.dark ? 0.11 : 0.125}
          sizeAttenuation
          vertexColors
          map={texture}
          alphaTest={0.02}
          transparent
          depthWrite={false}
          opacity={0.95}
          blending={palette.dark ? THREE.AdditiveBlending : THREE.NormalBlending}
        />
      </points>
      <gridHelper args={[16, 32, palette.dark ? "#3a1a16" : "#ead4b4", palette.dark ? "#241210" : "#f1e2c8"]} position={[0, -1.82, 0]} />
    </group>
  );
}

export default function HeroScene() {
  const host = useRef<HTMLDivElement>(null);
  const palette = usePalette();
  const still = useReducedMotionPref();
  const visible = useOnScreen(host);
  const [webgl, setWebgl] = useState(false);
  useEffect(() => setWebgl(hasWebGL()), []);

  return (
    <div ref={host} style={{ position: "absolute", inset: 0 }}>
      {webgl && palette ? (
        <Canvas
          camera={{ position: [0, 1.2, 9], fov: 45 }}
          dpr={[1, 1.75]}
          frameloop={visible && !still ? "always" : "demand"}
          gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
          onCreated={({ scene }) => {
            scene.fog = new THREE.Fog(palette.bg, palette.dark ? 8 : 10, palette.dark ? 17 : 22);
          }}
        >
          <Fog color={palette.bg} near={palette.dark ? 8 : 10} far={palette.dark ? 17 : 22} />
          <Cloud palette={palette} still={still} />
        </Canvas>
      ) : null}
    </div>
  );
}

function Fog({ color, near, far }: { color: string; near: number; far: number }) {
  const { scene } = useThree();
  useEffect(() => {
    scene.fog = new THREE.Fog(color, near, far);
  }, [scene, color, near, far]);
  return null;
}
