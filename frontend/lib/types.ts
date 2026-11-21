// The JSON shapes plainml's web API returns (plainml/web/server.py).

export type Kind = "numeric" | "categorical" | "datetime" | "text" | "id" | "constant" | "empty";

export interface Column {
  name: string;
  kind: Kind;
  missing_pct: number;
  unique: number;
  examples: string[];
  note?: string | null;
}

export interface Table {
  columns: string[];
  rows: (string | number | boolean | null)[][];
  total: number;
}

export interface Sample {
  key: string;
  file: string;
  title: string;
  question: string;
  about: string;
  rows: number;
  task: string;
  options: Record<string, unknown>;
}

export interface Upload {
  id: string;
  name: string;
  size: number;
  rows: number;
  columns: Column[];
  preview: Table;
  suggested_target: string | null;
  date_columns: string[];
  sample?: Sample | null;
}

export interface ImportanceMethod {
  key: string;
  label: string;
  about: string;
  default: boolean;
  family: string;
}

export interface Info {
  version: string;
  mode: "server" | "browser";
  needs_token: boolean;
  signed_in: boolean;
  runs_dir: string;
  upload_types: string[];
  importance_methods: ImportanceMethod[];
  max_upload_mb: number;
  samples?: Sample[];
  installed: Record<string, boolean>;
}

export interface Tile {
  label: string;
  value: string | number | null;
  sub?: string | null;
}

export interface FileEntry {
  path: string;
  name: string;
  label: string;
  description: string;
  size: number;
  type: string;
  url: string;
  download: string;
  preview: string | null;
}

export interface OutputResult {
  kind: "output";
  title: string;
  tiles: Tile[];
  files: FileEntry[];
  preview: Table | null;
  report: string | null;
}

export interface RunResult {
  kind: "run";
  run: string;
}

export interface Job {
  id: string;
  task: string;
  title: string;
  status: "queued" | "running" | "done" | "failed";
  stage: string;
  progress: number | null;
  elapsed: number;
  log: string[];
  log_size: number;
  result: OutputResult | RunResult | null;
  error: { message: string; hint?: string | null } | null;
}

export interface RunRow {
  run: string;
  kind: string;
  created?: string;
  data?: string;
  target?: string;
  best_model?: string;
  metric?: string;
  score?: number | null;
  [key: string]: unknown;
}

// --- data for the 3D views (run_visual in server.py) ---------------------------------------------

export interface Bar {
  label: string;
  value: number;
  tone?: "best" | "picked" | "warn" | "bad" | "plain";
  note?: string;
}

export type Visual =
  | {
      type: "bars";
      title: string;
      caption: string;
      value_label: string;
      bars: Bar[];
      lines?: { value: number; label: string }[];
      lower_is_better?: boolean;
    }
  | {
      type: "points";
      title: string;
      caption: string;
      x: number[];
      y: number[];
      z: number[];
      group: number[];
      groups: string[];
      size?: number[];
      hover?: string[];
    }
  | {
      type: "series";
      title: string;
      caption: string;
      value_label: string;
      lanes: {
        name: string;
        history: { x: number[]; y: number[] };
        forecast: { x: number[]; y: number[]; lower?: number[]; upper?: number[] };
      }[];
    };

export interface RunSummary {
  name: string;
  kind: string;
  title: string;
  subtitle: string;
  tiles: Tile[];
  sentences: string[];
  files: FileEntry[];
  report: string | null;
  predictable: boolean;
  visual?: Visual | null;
}
