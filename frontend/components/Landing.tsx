"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion, useScroll, useTransform, type MotionValue } from "motion/react";
import { api } from "@/lib/api";
import type { RunRow as Run, Sample, Upload } from "@/lib/types";
import { cx, num } from "@/lib/format";
import samples from "@/lib/samples.json";
import { useApp } from "./AppContext";
import { DropZone } from "./DropZone";
import { Icon } from "./Icon";
import { TASKS } from "./Options";
import { RunRow } from "./Runs";

const HeroScene = dynamic(() => import("./three/HeroScene"), { ssr: false });

function goTo(router: ReturnType<typeof useRouter>, upload: Upload) {
  router.push(`/analyze/?upload=${upload.id}`);
}

function Hero() {
  const { browser, boot } = useApp();
  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="hero-canvas" aria-hidden="true">
        <div className="hero-fallback" />
        <HeroScene />
      </div>
      <div className="hero-inner">
        <div>
          <div className="badges">
            <span className="badge gold">Free · open source · no sign-up</span>
            <span className="badge">
              <i className="live" />
              {browser ? (boot?.ready ? "Python is running in this tab" : "Python is loading in this tab") : "Running on your machine"}
            </span>
          </div>
          <h1 id="hero-title" className="display">
            <span>Upload a spreadsheet.</span>
            <span className="shine">Get a model you understand.</span>
          </h1>
          <p className="lead">
            plainml compares dozens of machine-learning models, tests the winner on rows it never saw, and tells you what it found in plain English.
            Forecasts, customer groups and unusual rows too.
          </p>
          <div className="hero-actions">
            <a className="btn primary big" href="#start">
              Start now <Icon name="arrow" />
            </a>
            <a className="btn big" href="https://github.com/pranay-obla/plainml">
              <Icon name="github" />
              Star on GitHub
            </a>
          </div>
          <p className="hero-note">
            <Icon name="lock" />
            {browser ? "Everything runs on your own computer. Your file is never uploaded anywhere." : "plainml is running on this machine. Your data stays here."}
          </p>
        </div>
      </div>
    </section>
  );
}

const STATEMENT = "Most AutoML hands you a score. plainml tells you what drives it, whether to trust it, and what to do next.";
const KEY_WORDS = new Set(["drives", "trust", "next."]);

function Word({ word, index, total, progress }: { word: string; index: number; total: number; progress: MotionValue<number> }) {
  const start = index / total;
  const opacity = useTransform(progress, [start, Math.min(1, start + 2 / total)], [0.16, 1]);
  return (
    <motion.span className={cx("w", KEY_WORDS.has(word) && "key")} style={{ opacity }}>
      {word}{" "}
    </motion.span>
  );
}

function Statement() {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 0.85", "end 0.45"] });
  const words = STATEMENT.split(" ");
  return (
    <div className="statement" ref={ref}>
      <p>
        {reduce
          ? words.map((w, i) => (
              <span key={i} className={cx("w", KEY_WORDS.has(w) && "key")}>
                {w}{" "}
              </span>
            ))
          : words.map((w, i) => <Word key={i} word={w} index={i} total={words.length} progress={scrollYProgress} />)}
      </p>
    </div>
  );
}

function Examples() {
  const router = useRouter();
  const { browser, boot, fail } = useApp();
  const [busy, setBusy] = useState<string | null>(null);
  const use = async (sample: Sample) => {
    if (busy) return;
    setBusy(sample.key);
    try {
      goTo(router, await api<Upload>(`/api/samples/${encodeURIComponent(sample.key)}`, { method: "POST" }));
    } catch (error) {
      fail(error);
      setBusy(null);
    }
  };
  const waiting = browser && !boot?.ready;
  return (
    <section className="band" id="start" style={{ scrollMarginTop: 70 }}>
      <div className="container">
        <Reveal className="band-head">
          <div>
            <div className="eyebrow">Start here</div>
            <h2 style={{ marginTop: 10 }}>Bring your data, or borrow ours.</h2>
          </div>
          <p>Drop in a file, or click an example: it loads made-up data and fills in the settings, so you just press Run.</p>
        </Reveal>
        <div className="start-grid">
          <div className="hero-side">
            <DropZone onFile={(upload) => goTo(router, upload)} />
          </div>
        <div className={cx("cards", "examples", busy && "busy")} role="list" id="examples">
          {(samples as Sample[]).map((sample, i) => {
            const task = TASKS.find((t) => t.key === sample.task) || TASKS[0];
            const mine = busy === sample.key;
            return (
              <Reveal key={sample.key} delay={i * 0.05}>
                <button type="button" role="listitem" className={cx("card", mine && "on")} onClick={() => use(sample)} title={sample.about} style={{ width: "100%", height: "100%" }}>
                  <div className="ico">{mine ? <span className="spin" aria-label="Loading" /> : <Icon name={task.icon} />}</div>
                  <Icon name="arrow" className="go" />
                  <b>{sample.title}</b>
                  <p>{sample.question}</p>
                  <span className="meta">
                    {mine && waiting ? "Waiting for Python to finish loading…" : `${task.title} · ${num(sample.rows)} rows`}
                  </span>
                </button>
              </Reveal>
            );
          })}
        </div>
        </div>
      </div>
    </section>
  );
}

function Reveal({ children, className, delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const reduce = useReducedMotion();
  if (reduce) return <div className={className}>{children}</div>;
  return (
    <motion.div className={className} initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }} transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}>
      {children}
    </motion.div>
  );
}

function HowItWorks() {
  return (
    <section className="band alt">
      <div className="container">
        <Reveal className="band-head">
          <div>
            <div className="eyebrow">How it works</div>
            <h2 style={{ marginTop: 10 }}>Three steps, no code.</h2>
          </div>
          <p>The same engine powers the command line and the Python API, so anything you do here you can repeat anywhere.</p>
        </Reveal>
        <div className="steps3">
          <Reveal className="step">
            <h3>Drop in your data</h3>
            <p>CSV, Excel, Parquet or JSON. plainml reads messy exports: prices stored as text, day-first dates, blanks and free-text notes.</p>
          </Reveal>
          <Reveal className="step" delay={0.08}>
            <h3>Ask a question</h3>
            <p>Predict a column, forecast, find groups or unusual rows, rank columns or check for drift. Sensible settings are filled in.</p>
          </Reveal>
          <Reveal className="step" delay={0.16}>
            <h3>Read the answer</h3>
            <p>Headline numbers, findings in plain English and a full report, with every file ready to download.</p>
            <div className="quote">&ldquo;Higher support_calls → churn more likely (24% → 38%).&rdquo;</div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

const PROMISES = [
  { icon: "scale", title: "Beats a baseline", text: "Every model is compared with a do-nothing guess. If nothing beats it, you're told." },
  { icon: "check", title: "Tested fairly", text: "Cross-validation to choose, then a final test on rows the model never saw." },
  { icon: "alert", title: "Warns about leaks", text: "ID columns, target leaks and imbalanced classes are caught before training." },
  { icon: "chat", title: "Explains itself", text: "Which columns matter, how they move the prediction, and whether to trust its probabilities." },
] as const;

function Promises() {
  return (
    <section className="band">
      <div className="container">
        <Reveal className="band-head">
          <div>
            <div className="eyebrow">Honest by default</div>
            <h2 style={{ marginTop: 10 }}>A model you can trust, not just a number.</h2>
          </div>
        </Reveal>
        <div className="promise">
          {PROMISES.map((p, i) => (
            <Reveal key={p.title} delay={i * 0.06}>
              <Icon name={p.icon} />
              <b>{p.title}</b>
              <p>{p.text}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

function RecentRuns() {
  const { info } = useApp();
  const [runs, setRuns] = useState<Run[]>([]);
  useEffect(() => {
    if (info) api<Run[]>("/api/runs").then(setRuns).catch(() => setRuns([]));
  }, [info]);
  if (!runs.length) return null;
  return (
    <section className="band alt">
      <div className="container">
        <div className="section-head">
          <h2>Your recent results</h2>
          <Link href="/runs/">All runs →</Link>
        </div>
        <div className="runs">
          {runs.slice(0, 4).map((r) => (
            <RunRow key={r.run} run={r} />
          ))}
        </div>
      </div>
    </section>
  );
}

export function Landing() {
  return (
    <main className="page flush">
      <Hero />
      <Examples />
      <RecentRuns />
      <Statement />
      <HowItWorks />
      <Promises />
    </main>
  );
}
