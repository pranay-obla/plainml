"use client";

// A real plainml 3D view on the home page: the app's own scene (components/three/Scenes), loaded when it
// nears the screen. Without WebGL it shows the still capture of the same view.

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
}: {
  visual: Visual | null;
  tone?: Tone;
  spin?: boolean;
  label: string;
  still: string;
  className?: string;
  far?: number;
  forceStill?: boolean; // the data could not load: show the still capture instead
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [gl, setGl] = useState<boolean | null>(null);
  const [near, setNear] = useState(false);

  useEffect(() => {
    setGl(hasWebGL());
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

  const live = gl && near && visual && !forceStill;
  const showStill = gl === false || forceStill;
  return (
    <div ref={ref} className={className ? `stage ${className}` : "stage"} role="img" aria-label={label}>
      {live ? <Scenes visual={visual} tone={tone} spin={spin} far={far} /> : showStill ? <img className="stage__still" src={still} alt="" /> : null}
    </div>
  );
}
