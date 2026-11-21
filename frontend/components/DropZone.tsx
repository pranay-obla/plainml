"use client";

import { useRef, useState, type DragEvent, type KeyboardEvent } from "react";
import { uploadFile } from "@/lib/api";
import type { Upload } from "@/lib/types";
import { cx } from "@/lib/format";
import { useApp } from "./AppContext";
import { Icon } from "./Icon";

const FALLBACK_TYPES = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".feather"];

export function DropZone({
  onFile,
  label = "Drop a data file here",
  compact = false,
}: {
  onFile: (upload: Upload) => void | Promise<void>;
  label?: string;
  compact?: boolean;
}) {
  const { info, browser, boot, fail } = useApp();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [share, setShare] = useState(0);

  const take = async (file: File | undefined) => {
    if (!file || busy) return;
    setBusy(browser && !boot?.ready ? `Setting up Python, then reading ${file.name}…` : `Uploading ${file.name}…`);
    setShare(0);
    try {
      const upload = await uploadFile(file, (value) => {
        setShare(value);
        if (value >= 1) setBusy(`Reading ${file.name}…`);
      });
      await onFile(upload);
    } catch (error) {
      fail(error);
      setBusy(null);
    }
  };

  const keyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.current?.click();
    }
  };
  const drop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    take(e.dataTransfer.files[0]);
  };

  return (
    <div
      className={cx("drop", compact && "compact", over && "over")}
      role="button"
      tabIndex={0}
      aria-label={`${label}, or press Enter to browse`}
      onClick={() => input.current?.click()}
      onKeyDown={keyDown}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <div className="orb">
        <Icon name="upload" />
      </div>
      <strong>{busy || label}</strong>
      {busy ? (
        <div className="upbar" aria-hidden="true">
          <i style={{ width: `${Math.round(share * 100)}%` }} />
        </div>
      ) : (
        <>
          <div className="sub">
            or <span style={{ color: "var(--accent-text)", fontWeight: 650 }}>browse your files</span>
          </div>
          <div className="formats">CSV, Excel, Parquet, JSON, TSV · up to {info?.max_upload_mb ?? (browser ? 200 : 500)} MB</div>
        </>
      )}
      <input
        ref={input}
        type="file"
        accept={(info?.upload_types || FALLBACK_TYPES).join(",")}
        onChange={(e) => take(e.target.files?.[0])}
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
