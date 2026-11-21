export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function num(value: unknown): string {
  if (value === null || value === undefined || value === "") return "";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Number.isInteger(n)) return n.toLocaleString();
  const size = Math.abs(n);
  const digits = size >= 1000 ? 0 : size >= 100 ? 1 : size >= 1 ? 2 : 4;
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function duration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

export const KIND_LABEL: Record<string, string> = {
  numeric: "number",
  categorical: "category",
  datetime: "date",
  text: "text",
  id: "ID",
  constant: "constant",
  empty: "empty",
};

export const METRIC_LABELS: Record<string, string> = {
  f1: "F1",
  f1_macro: "F1 (macro)",
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  roc_auc: "ROC-AUC",
  log_loss: "Log loss",
  precision: "Precision",
  recall: "Recall",
  r2: "R²",
  rmse: "RMSE",
  mae: "MAE",
  mape: "MAPE",
  silhouette: "Silhouette",
  hamming: "Hamming loss",
  average_precision: "Avg. precision",
};

export function cx(...names: (string | false | null | undefined)[]): string {
  return names.filter(Boolean).join(" ");
}
