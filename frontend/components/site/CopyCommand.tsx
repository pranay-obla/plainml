"use client";

import { useState } from "react";

/** A shell command as a button: one click copies it. */
export function CopyCommand({ command, tone }: { command: string; tone?: "band" }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setState("copied");
    } catch {
      setState("failed");
    }
    window.setTimeout(() => setState("idle"), 1800);
  };
  return (
    <button type="button" className={tone ? `cmd cmd--${tone}` : "cmd"} onClick={copy} data-state={state}>
      <code>{command}</code>
      <span className="cmd__state" aria-live="polite">
        {state === "copied" ? "Copied" : state === "failed" ? "Select it to copy" : "Copy"}
      </span>
    </button>
  );
}
