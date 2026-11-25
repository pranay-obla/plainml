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
}: {
  visual: Visual | null;
  tone?: Tone;
  spin?: boolean;
  label: string;
  still: string;
  className?: string;
  far?: number;
  forceStill?: boolean; // the data could not load: keep the still capture
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [gl, setGl] = useState<boolean | null>(null);
  const [near, setNear] = useState(false);
  const [shown, setShown] = useState(false);
  const [paused, setPaused] = useState(false);

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

  const live = Boolean(gl && near && visual && !forceStill);
  useEffect(() => {
    if (!live) return;
    // the canvas paints within a frame or two of mounting; then the still fades out under it
    const t = window.setTimeout(() => setShown(true), 450);
    return () => window.clearTimeout(t);
  }, [live]);

  return (
    <div ref={ref} className={className ? `stage ${className}` : "stage"} data-shown={live && shown}>
      <div className="stage__view" role="img" aria-label={label}>
        <img className="stage__still" src={still} alt="" />
        {live ? (
          <div className="stage__live">
            <Scenes visual={visual as Visual} tone={tone} spin={spin && !paused} far={far} />
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
