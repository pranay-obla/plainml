"use client";

// A real plainml 3D view on the home page: the app's own scene (components/three/Scenes). The still
// capture of the same view shows first, server-rendered, and stays when WebGL or the data is missing;
// the live scene loads when the stage nears the screen and fades in over it.

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type { Visual } from "@/lib/types";
import { hasWebGL, type Tone } from "../three/theme";

const Scenes = dynamic(() => import("../three/Scenes"), { ssr: false });

export function SceneStage({
  visual,
  tone,
  spin,
  label,
  still,
  className,
  far,
  forceStill,
  grow,
  floor,
}: {
  visual: Visual | null;
  tone?: Tone;
  spin?: boolean;
  label: string;
  still: string;
  className?: string;
  far?: number;
  forceStill?: boolean; // the data could not load: keep the still capture
  grow?: boolean; // the bars rise from the floor on arrival (the hero); the still stays hidden meanwhile
  floor?: string; // with grow: a still of the scene's first frame (the empty floor), shown until the canvas is
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [gl, setGl] = useState<boolean | null>(null);
  const [near, setNear] = useState(false);
  const [shown, setShown] = useState(false);
  const [paused, setPaused] = useState(false);
  const [reduce, setReduce] = useState(false);
  const [ready, setReady] = useState(false); // the canvas exists
  const [late, setLate] = useState(false); // too slow to grow: show the still and cross-fade instead
  const stillRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    setGl(hasWebGL());
    setReduce(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    // the still already came back (the CSS fallback, on a slow load): never hide it again to grow
    if (stillRef.current && parseFloat(getComputedStyle(stillRef.current).opacity) > 0.1) setLate(true);
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setNear(true);
          io.disconnect();
        }
      },
      { rootMargin: "400px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const live = Boolean(gl && near && visual && !forceStill);
  const growing = Boolean(grow && !reduce && !late);
  useEffect(() => {
    // three.js arrives in its own chunk; if it takes longer than this, the still comes back instead
    if (!growing || ready || gl !== true) return;
    const t = window.setTimeout(() => setLate(true), 1200);
    return () => window.clearTimeout(t);
  }, [growing, ready, gl]);
  useEffect(() => {
    if (!live) return;
    // a growing scene starts from an empty floor, so it shows as soon as the canvas exists; otherwise the
    // canvas paints within a frame or two of being created, then the still fades out under it
    if (!ready) return; // three.js can be slow to arrive: the still stays until the canvas exists
    if (growing) {
      // the floor still is this scene's first frame, so the canvas replaces it outright, two frames in
      let f = requestAnimationFrame(() => {
        f = requestAnimationFrame(() => setShown(true));
      });
      return () => cancelAnimationFrame(f);
    }
    const t = window.setTimeout(() => setShown(true), 200);
    return () => window.clearTimeout(t);
  }, [live, growing, ready]);
  // what the CSS needs before and after the check: "grow" keeps the still hidden, "still" brings it back
  const glState = gl === null ? undefined : gl && !forceStill ? (growing ? "grow" : "live") : "still";

  return (
    <div
      ref={ref}
      className={["stage", className, grow ? "stage--grow" : ""].filter(Boolean).join(" ")}
      data-shown={live && shown}
      data-gl={glState}
    >
      <div className="stage__view" role="img" aria-label={label}>
        {grow && floor ? <img className="stage__floor" src={floor} alt="" /> : null}
        <img ref={stillRef} className="stage__still" src={still} alt="" />
        {live ? (
          <div className="stage__live">
            <Scenes visual={visual as Visual} tone={tone} spin={spin && !paused} far={far} grow={growing} begin={shown} onReady={() => setReady(true)} />
          </div>
        ) : null}
      </div>
      {live && spin ? (
        <button type="button" className="stage__pause" aria-pressed={paused} onClick={() => setPaused((p) => !p)}>
          {paused ? "Turn" : "Pause"}
        </button>
      ) : null}
    </div>
  );
}
