"use client";

import { useEffect, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import type { Column, Info, RunRow, Upload } from "@/lib/types";
import { cx, KIND_LABEL, num } from "@/lib/format";
import { DropZone } from "./DropZone";
import { Icon, type IconName } from "./Icon";

export interface Task {
  key: string;
  title: string;
  about: string;
  icon: IconName;
}

export const TASKS: Task[] = [
  { key: "train", title: "Predict a column", about: "Train and compare models that predict one column from the others.", icon: "target" },
  { key: "forecast", title: "Forecast over time", about: "Project a value forward from its history, with a likely range.", icon: "trend" },
  { key: "cluster", title: "Find groups", about: "Split rows into natural segments and describe what sets each apart.", icon: "groups" },
  { key: "anomaly", title: "Find unusual rows", about: "Score every row and flag the ones that don't fit the pattern.", icon: "alert" },
  { key: "importance", title: "Rank the columns", about: "See which columns matter for a target, compared across many methods.", icon: "bars" },
  { key: "drift", title: "Check for drift", about: "Compare this data with a model's training data or another file.", icon: "drift" },
  { key: "profile", title: "Profile the data", about: "Summarise every column and flag problems before modelling.", icon: "scan" },
  { key: "clean", title: "Clean the data", about: "Fix types, blanks, duplicates and odd values, then download it.", icon: "broom" },
];

export type Options = Record<string, unknown>;
type Set = (name: string, value: unknown) => void;

function columnsOf(upload: Upload, kinds?: string[]): Column[] {
  return upload.columns.filter((c) => !kinds || kinds.includes(c.kind));
}

function pickColumn(upload: Upload, kinds: string[] | undefined, wanted: unknown): string | undefined {
  const choices = columnsOf(upload, kinds);
  if (typeof wanted === "string" && choices.some((c) => c.name === wanted)) return wanted;
  return undefined;
}

/** The starting settings for a task: the example dataset's presets, else sensible defaults. */
export function defaultsFor(task: string, upload: Upload, info: Info): Options {
  const preset = upload.sample && upload.sample.task === task ? upload.sample.options : {};
  const suggested = (preset.target as string) || upload.suggested_target;
  const first = (kinds: string[]) => columnsOf(upload, kinds)[0]?.name ?? "";
  switch (task) {
    case "train":
      return {
        target: pickColumn(upload, ["numeric", "categorical", "text"], suggested) ?? first(["numeric", "categorical", "text"]),
        speed: preset.speed ?? "standard",
        drop: [],
      };
    case "forecast":
      return {
        target: pickColumn(upload, ["numeric"], suggested) ?? first(["numeric"]),
        horizon: preset.horizon !== undefined ? String(preset.horizon) : "",
        group: pickColumn(upload, ["categorical"], preset.group) ?? "",
        inputs: ((preset.inputs as string[]) || []).filter((c) => pickColumn(upload, ["numeric"], c)),
        country: info.installed.holidays ? ((preset.country as string) ?? "") : "",
        agg: "sum",
        date: "",
      };
    case "cluster":
      return { k: "", drop: [] };
    case "anomaly":
      return { contamination: "", label: pickColumn(upload, ["categorical", "numeric"], preset.label) ?? "", drop: [] };
    case "importance":
      return {
        target: pickColumn(upload, ["numeric", "categorical"], suggested) ?? first(["numeric", "categorical"]),
        k: "",
        methods: info.importance_methods.filter((m) => m.default).map((m) => m.key),
        drop: [],
      };
    case "drift":
      return { reference_mode: "model", target: "" };
    case "profile":
      return { target: "" };
    case "clean":
      return { target: "", outliers: "none", drop_duplicates: true, impute: false, encode: false };
    default:
      return {};
  }
}

export function cleanOptions(options: Options): Options {
  const out: Options = {};
  for (const [key, value] of Object.entries(options)) {
    if (key === "reference_mode" || value === "" || value === null || value === undefined) continue;
    if (Array.isArray(value) && !value.length) continue;
    out[key] = value;
  }
  return out;
}

// --- controls ---------------------------------------------------------------------------------------

let ids = 0;
function Field({ label, hint, wide, children, htmlFor }: { label: string; hint?: ReactNode; wide?: boolean; children: ReactNode; htmlFor?: string }) {
  return (
    <div className={cx("field", wide && "wide")}>
      {htmlFor ? <label htmlFor={htmlFor}>{label}</label> : <div className="label">{label}</div>}
      {children}
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

function useId() {
  const [id] = useState(() => `f-${++ids}`);
  return id;
}

function SelectField(props: { label: string; hint?: string; name: string; value: unknown; set: Set; choices: { value: string; label: string }[]; blank?: string }) {
  const id = useId();
  return (
    <Field label={props.label} hint={props.hint} htmlFor={id}>
      <select id={id} className="input" value={String(props.value ?? "")} onChange={(e) => props.set(props.name, e.target.value)}>
        {props.blank !== undefined ? <option value="">{props.blank}</option> : null}
        {props.choices.map((c) => (
          <option key={c.value} value={c.value}>
            {c.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

function columnChoices(upload: Upload, kinds?: string[]) {
  return columnsOf(upload, kinds).map((c) => ({ value: c.name, label: `${c.name}  ·  ${KIND_LABEL[c.kind] || c.kind}` }));
}

function TextField(props: { label: string; hint?: string; name: string; value: unknown; set: Set; placeholder: string; type?: string }) {
  const id = useId();
  return (
    <Field label={props.label} hint={props.hint} htmlFor={id}>
      <input
        id={id}
        className="input"
        type={props.type || "text"}
        inputMode={props.type === "number" ? "decimal" : undefined}
        placeholder={props.placeholder}
        value={String(props.value ?? "")}
        onChange={(e) => props.set(props.name, e.target.value)}
      />
    </Field>
  );
}

export function Segmented({ value, choices, onChange, label }: { value: string; choices: { value: string; label: string }[]; onChange: (v: string) => void; label?: string }) {
  return (
    <div className="seg" role="radiogroup" aria-label={label}>
      {choices.map((c) => (
        <button key={c.value} type="button" role="radio" aria-checked={c.value === value} className={cx(c.value === value && "on")} onClick={() => onChange(c.value)}>
          {c.label}
        </button>
      ))}
    </div>
  );
}

function Chips({ name, value, set, choices }: { name: string; value: unknown; set: Set; choices: { value: string; label: string; title?: string }[] }) {
  const list = (value as string[]) || [];
  return (
    <div className="chips" role="group">
      {choices.map((c) => {
        const on = list.includes(c.value);
        return (
          <button
            key={c.value}
            type="button"
            className={cx("chip", on && "on")}
            aria-pressed={on}
            title={c.title}
            onClick={() => set(name, on ? list.filter((v) => v !== c.value) : [...list, c.value])}
          >
            {c.label}
          </button>
        );
      })}
    </div>
  );
}

function Toggle({ name, value, set, label, about }: { name: string; value: unknown; set: Set; label: string; about?: string }) {
  return (
    <label className="switch">
      <input type="checkbox" role="switch" checked={Boolean(value)} onChange={(e) => set(name, e.target.checked)} />
      <div>
        <b>{label}</b>
        {about ? <span>{about}</span> : null}
      </div>
    </label>
  );
}

function Advanced({ children, open }: { children: ReactNode; open?: boolean }) {
  return (
    <details className="more field wide" open={open}>
      <summary>More options</summary>
      <div className="form" style={{ marginTop: 16 }}>
        {children}
      </div>
    </details>
  );
}

function DropColumns({ upload, options, set }: { upload: Upload; options: Options; set: Set }) {
  return (
    <Field label="Columns to ignore" hint="Leave out IDs, leaks, or anything you won't know when predicting." wide>
      <Chips name="drop" value={options.drop} set={set} choices={columnsOf(upload).map((c) => ({ value: c.name, label: c.name }))} />
    </Field>
  );
}

const PRIVATE = "Keep raw data values out of the report and saved files.";

function DriftReference({ options, set }: { options: Options; set: Set }) {
  const [models, setModels] = useState<RunRow[] | null>(null);
  const [file, setFile] = useState<Upload | null>(null);
  useEffect(() => {
    api<RunRow[]>("/api/models")
      .then((all) => {
        const trained = all.filter((m) => ["train", "tune"].includes(m.kind));
        setModels(trained);
        if (trained.length && !options.reference_run && options.reference_mode === "model") set("reference_run", trained[0].run);
      })
      .catch(() => setModels([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const mode = String(options.reference_mode || "model");
  const choose = (value: string) => {
    set("reference_mode", value);
    if (value === "model") {
      set("reference_upload", null);
      if (models?.length) set("reference_run", models[0].run);
    } else set("reference_run", null);
  };
  return (
    <>
      <Field label="Compare with" wide>
        <Segmented value={mode} onChange={choose} label="Compare with" choices={[{ value: "model", label: "A trained model" }, { value: "file", label: "Another file" }]} />
      </Field>
      {mode === "model" ? (
        models === null ? (
          <div className="field wide muted">Loading models…</div>
        ) : models.length ? (
          <div className="field wide">
            <SelectField
              label="Model"
              hint="The new data is compared with the data this model learned from."
              name="reference_run"
              value={options.reference_run}
              set={set}
              choices={models.map((m) => ({ value: m.run, label: `${m.target} · ${m.best_model} · ${m.created}` }))}
            />
          </div>
        ) : (
          <p className="sub field wide">No trained models yet. Compare with another file instead.</p>
        )
      ) : file ? (
        <Field label="Reference file" wide>
          <span className="pill">
            <Icon name="table" />
            <b>{file.name}</b> {num(file.rows)} rows
          </span>
        </Field>
      ) : (
        <Field label={'Reference file (the "before" data)'} wide>
          <DropZone
            compact
            label="Drop the reference file here"
            onFile={(upload) => {
              setFile(upload);
              set("reference_upload", upload.id);
            }}
          />
        </Field>
      )}
    </>
  );
}

export function OptionsForm({ task, upload, info, options, set }: { task: string; upload: Upload; info: Info; options: Options; set: Set }) {
  const p = { set };
  switch (task) {
    case "train":
      return (
        <>
          <SelectField label="Column to predict" hint="Numbers give a regression; labels give a classification." name="target" value={options.target} choices={columnChoices(upload, ["numeric", "categorical", "text"])} {...p} />
          <Field label="How hard to try" hint={`Quick tries fast models only. Thorough adds more models${info.installed.torch ? " (including a neural network)" : ""} and a stacked ensemble.`}>
            <Segmented value={String(options.speed)} onChange={(v) => set("speed", v)} label="How hard to try" choices={[{ value: "quick", label: "Quick" }, { value: "standard", label: "Standard" }, { value: "thorough", label: "Thorough" }]} />
          </Field>
          <Advanced>
            <TextField label="Time limit" hint="Stop starting new models after this long." name="time_budget" value={options.time_budget} placeholder="e.g. 5m or 1h" {...p} />
            <DropColumns upload={upload} options={options} set={set} />
            <Toggle name="calibrate" value={options.calibrate} label="Calibrate probabilities" about="So '70% sure' means right 70% of the time." {...p} />
            <Toggle name="log_target" value={options.log_target} label="Model the log of the target" about="Helps with skewed positive amounts like prices." {...p} />
            <Toggle name="private" value={options.private} label="Private" about={PRIVATE} {...p} />
          </Advanced>
        </>
      );
    case "forecast":
      return (
        <>
          <SelectField label="Value to forecast" name="target" value={options.target} choices={columnChoices(upload, ["numeric"])} {...p} />
          <SelectField label="Date column" name="date" value={options.date} choices={columnChoices(upload, ["datetime"])} blank="Detect it for me" {...p} />
          <TextField label="How far ahead" hint="In periods of the data's frequency (days, weeks, months…)." name="horizon" value={options.horizon} placeholder="Automatic" type="number" {...p} />
          <SelectField label="One forecast per" hint="E.g. a store or product column." name="group" value={options.group} choices={columnChoices(upload, ["categorical"])} blank="No groups (one series)" {...p} />
          <Advanced open={Boolean((options.inputs as string[])?.length || options.country)}>
            <Field label="Inputs known in advance" hint="Promotions, prices… Rows after the last known value can hold their planned values." wide>
              <Chips name="inputs" value={options.inputs} set={set} choices={columnsOf(upload, ["numeric"]).map((c) => ({ value: c.name, label: c.name }))} />
            </Field>
            <TextField label="Public holidays" name="country" value={options.country} placeholder={info.installed.holidays ? "Country code, e.g. US, GB, IN" : "Needs: pip install holidays"} {...p} />
            <SelectField label="Combine rows on the same date by" name="agg" value={options.agg} choices={[{ value: "sum", label: "Sum" }, { value: "mean", label: "Average" }, { value: "last", label: "Last value" }]} {...p} />
          </Advanced>
        </>
      );
    case "cluster":
      return (
        <>
          <TextField label="Number of groups" hint="Leave blank to let plainml choose." name="k" value={options.k} placeholder="Find the best number" type="number" {...p} />
          <Advanced>
            <DropColumns upload={upload} options={options} set={set} />
            <Toggle name="private" value={options.private} label="Private" about={PRIVATE} {...p} />
          </Advanced>
        </>
      );
    case "anomaly":
      return (
        <>
          <TextField label="Expected share of unusual rows (%)" hint="E.g. 1 for about one row in a hundred." name="contamination" value={options.contamination} placeholder="Estimate it for me" type="number" {...p} />
          <SelectField label="Known labels (optional)" hint="A column marking known anomalies, to check the results against." name="label" value={options.label} choices={columnChoices(upload, ["categorical", "numeric"])} blank="None" {...p} />
          <Advanced>
            <DropColumns upload={upload} options={options} set={set} />
            <Toggle name="private" value={options.private} label="Private" about={PRIVATE} {...p} />
          </Advanced>
        </>
      );
    case "importance":
      return (
        <>
          <SelectField label="Column the others should predict" name="target" value={options.target} choices={columnChoices(upload, ["numeric", "categorical"])} {...p} />
          <TextField label="How many columns to keep" name="k" value={options.k} placeholder="Decide from the data" type="number" {...p} />
          <Field label="Methods" hint="The defaults balance speed and depth. Hover over a method to see what it does." wide>
            <Chips name="methods" value={options.methods} set={set} choices={info.importance_methods.map((m) => ({ value: m.key, label: m.label, title: m.about }))} />
          </Field>
          <Advanced>
            <DropColumns upload={upload} options={options} set={set} />
            <Toggle name="private" value={options.private} label="Private" about={PRIVATE} {...p} />
          </Advanced>
        </>
      );
    case "drift":
      return (
        <>
          <DriftReference options={options} set={set} />
          <SelectField label="Target column (optional)" hint="Leave it out of the comparison, since it's what gets predicted." name="target" value={options.target} choices={columnChoices(upload)} blank="None" {...p} />
        </>
      );
    case "profile":
      return <SelectField label="Target column (optional)" hint="Adds class balance and leakage checks." name="target" value={options.target} choices={columnChoices(upload, ["numeric", "categorical"])} blank="None" {...p} />;
    case "clean":
      return (
        <>
          <SelectField label="Target column (optional)" hint="Rows without a target are dropped; it's never imputed or encoded." name="target" value={options.target} choices={columnChoices(upload)} blank="None" {...p} />
          <SelectField label="Outliers" name="outliers" value={options.outliers} choices={[{ value: "none", label: "Leave them" }, { value: "clip", label: "Clip to a sensible range" }, { value: "remove", label: "Remove those rows" }]} {...p} />
          <Toggle name="drop_duplicates" value={options.drop_duplicates} label="Remove duplicate rows" {...p} />
          <Toggle name="impute" value={options.impute} label="Fill in blanks" about="Numbers with the median, categories with the most common value." {...p} />
          <Toggle name="encode" value={options.encode} label="Encode categories as numbers" about="One column per category, ready for other tools." {...p} />
        </>
      );
    default:
      return null;
  }
}
