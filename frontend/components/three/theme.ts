"use client";

import { useEffect, useState } from "react";

// Categorical colors for groups, in a fixed order (checked for color-blind separation against
// each theme's surface). A seventh group onwards shares "other".
const SERIES = {
  light: ["#b3261e", "#2a78d6", "#c98500", "#1baf7a", "#6250d6", "#e87ba4"],
  dark: ["#e66767", "#3987e5", "#c98500", "#199e70", "#9085e9", "#d55181"],
};

export interface Palette {
  dark: boolean;
  bg: string;
  surface: string;
  ink: string;
  ink2: string;
  muted: string;
  line: string;
  highlight: string; // the one bar or point that matters most
  gold: string;
  warn: string;
  bad: string;
  series: string[];
  other: string;
}

function isDark(): boolean {
  const theme = document.documentElement.dataset.theme;
  if (theme) return theme === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export type Tone = "light" | "dark" | "band";

function palette(tone?: Tone): Palette {
  // "band": drawn on the home page's cardinal hero, cream bars and a gold winner on a deeper red floor
  if (tone === "band")
    return { dark: true, bg: "#990000", surface: "#7a0000", ink: "#fff6e3", ink2: "#f6d9c8", muted: "#d9a493", line: "#6a0000", highlight: "#ffcc00", gold: "#ffcc00", warn: "#ffb020", bad: "#ffb4a8", series: SERIES.dark, other: "#f3d9c9" };
  const dark = tone ? tone === "dark" : isDark();
  return dark
    ? { dark, bg: "#0c0505", surface: "#1d0e0e", ink: "#fff6e3", ink2: "#cdb9aa", muted: "#6e5a52", line: "#3a2420", highlight: "#ffcc00", gold: "#ffcc00", warn: "#ffb020", bad: "#ff6b5e", series: SERIES.dark, other: "#8a7a72" }
    : { dark, bg: "#fff9ec", surface: "#fbf4e6", ink: "#1d0808", ink2: "#5f4747", muted: "#c9b8a6", line: "#e6d8c2", highlight: "#990000", gold: "#e0a800", warn: "#9a5b00", bad: "#b3261e", series: SERIES.light, other: "#a39588" };
}

export function seriesColor(p: Palette, index: number): string {
  return index < p.series.length ? p.series[index] : p.other;
}

/** The current theme's colors, updated when the theme changes; `tone` fixes them (a scene drawn on a
 *  coloured ground, like the home page's cardinal hero). */
export function usePalette(tone?: Tone): Palette | null {
  const [value, setValue] = useState<Palette | null>(null);
  useEffect(() => {
    const update = () => setValue(palette(tone));
    update();
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", update);
    window.addEventListener("plainml-theme", update);
    return () => {
      media.removeEventListener("change", update);
      window.removeEventListener("plainml-theme", update);
    };
  }, [tone]);
  return value;
}

export function useReducedMotionPref(): boolean {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduce(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return reduce;
}

export function hasWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Whether an element is on screen, so scenes stop drawing when scrolled away. */
export function useOnScreen(ref: React.RefObject<Element | null>): boolean {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: "100px" });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref]);
  return visible;
}
