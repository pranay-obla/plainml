"""``plainml export``: convert a model to ONNX so it runs outside Python.

ONNX models run in C++, C#, Java, JavaScript (onnxruntime-web) and more. The export
covers scikit-learn models (not XGBoost/LightGBM/CatBoost or ensembles). Two parts of
plainml's preprocessing need translating first, and the result is always checked
against the original model before it's saved:

- blanks in number columns: the "was this blank?" indicator columns get a small custom
  ONNX converter (skl2onnx doesn't ship one)
- blanks in category columns: ONNX string tensors can't hold NaN, so a blank category is
  sent as an empty string ""

Date columns become their parts (``signup__year``, ``signup__month``...), exactly as in
training. A ``.json`` file saved next to the model lists every input.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plainml.console import esc, fmt_num, info, note, success, warn
from plainml.errors import PlainMLError, require
from plainml.preprocessing import MISSING_CATEGORY, ColumnPrep, inner_pipeline, preprocess_step
from plainml.runs import DEFAULT_RUNS_DIR, HOLDOUT_FILE
from plainml.tasks import CLASSIFICATION, REGRESSION

TARGET_OPSET = 17
_registered = False


def _register_missing_indicator() -> None:
    """Teach skl2onnx to convert MissingIndicator: pick the columns, IsNaN, cast to float."""
    global _registered
    if _registered:
        return
    import onnx
    from skl2onnx import update_registered_converter
    from skl2onnx.algebra.onnx_ops import OnnxCast, OnnxGather, OnnxIsNaN
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.impute import MissingIndicator

    def shape(operator: Any) -> None:
        rows = operator.inputs[0].get_first_dimension()
        operator.outputs[0].type = FloatTensorType([rows, len(operator.raw_operator.features_)])

    def convert(scope: Any, operator: Any, container: Any) -> None:
        version = container.target_opset
        columns = np.asarray(operator.raw_operator.features_, dtype=np.int64)
        picked = OnnxGather(operator.inputs[0], columns, axis=1, op_version=version)
        missing = OnnxIsNaN(picked, op_version=version)
        OnnxCast(
            missing,
            to=onnx.TensorProto.FLOAT,
            op_version=version,
            output_names=[operator.outputs[0]],
        ).add_to(scope, container)

    update_registered_converter(MissingIndicator, "PlainMLMissingIndicator", shape, convert)
    _registered = True


def _onnx_friendly(encode: Any) -> Any:
    """An equivalent copy of the encoding step made only of ONNX-convertible parts."""
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.pipeline import FeatureUnion, Pipeline

    encode = copy.deepcopy(encode)
    rebuilt = []
    for name, transformer, columns in encode.transformers_:
        if name == "num":
            imputer = transformer.named_steps["impute"]
            if imputer.indicator_ is not None:
                plain = copy.deepcopy(imputer)
                plain.add_indicator = False
                plain.indicator_ = None
                pattern = np.zeros((2, len(columns)))
                pattern[0, imputer.indicator_.features_] = (
                    np.nan
                )  # same columns flagged as in training
                indicator = MissingIndicator(features="missing-only").fit(pattern)
                union = FeatureUnion([("impute", plain), ("missing", indicator)])
                transformer = Pipeline(
                    [("impute", union), ("scale", transformer.named_steps["scale"])]
                )
        elif name == "cat":
            blank = SimpleImputer(
                missing_values="", strategy="constant", fill_value=MISSING_CATEGORY
            )
            blank.fit(np.full((1, len(columns)), "x", dtype=object))
            transformer = Pipeline(
                [("impute", blank), ("onehot", transformer.named_steps["onehot"])]
            )
        elif name.startswith("text"):
            raise PlainMLError("Models that use free-text columns can't be exported to ONNX yet.")
        rebuilt.append((name, transformer, columns))
    encode.transformers_ = rebuilt
    return encode


def onnx_inputs(
    prep: ColumnPrep, frame: pd.DataFrame, names: dict[str, str] | None = None
) -> dict[str, np.ndarray]:
    """Turn raw rows into the ONNX model's inputs (one [n, 1] tensor per prepared column).

    ``names`` maps prepared columns to ONNX input names (skl2onnx renames 'a b' to 'a_b').
    """
    names = names or {}
    prepared = prep.transform(frame)
    feeds = {
        names.get(c, c): prepared[c].to_numpy(dtype=np.float32).reshape(-1, 1)
        for c in prep.numeric_out
    }
    for column in prep.categorical:
        values = prepared[column].astype(object)
        feeds[names.get(column, column)] = (
            values.where(values.notna(), "").astype(str).to_numpy().reshape(-1, 1)
        )
    return feeds


def export_onnx(
    model: Any = "latest",
    *,
    output: str | Path | None = None,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    verbose: bool = True,
) -> Path:
    """Export a classification or regression model to ONNX and verify it. Returns the path."""
    from plainml.console import quiet
    from plainml.predicting import load_model

    skl2onnx = require("skl2onnx", "ONNX export")
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    from sklearn.pipeline import Pipeline

    fitted, meta, model_path = load_model(model, out_dir=out_dir, with_path=True)
    task = meta.get("task")
    if task not in (CLASSIFICATION, REGRESSION):
        raise PlainMLError(
            f"ONNX export supports classification and regression models; this is a {task} model."
        )
    pipeline = inner_pipeline(fitted)
    if pipeline is None:
        raise PlainMLError(
            "Ensemble models can't be exported to ONNX.",
            hint="Retrain without the ensemble (--no-ensemble) or pick one model with --models.",
        )
    estimator = pipeline.named_steps["model"]
    if type(estimator).__module__.split(".")[0] not in ("sklearn",):
        raise PlainMLError(
            f"{meta.get('model', type(estimator).__name__)} models can't be exported to ONNX yet.",
            hint="Export works for scikit-learn models: retrain with e.g. --models rf,extratrees,linear",
        )
    if "balance" in pipeline.named_steps:
        raise PlainMLError(
            "Models trained with SMOTE can't be exported to ONNX yet.",
            hint="Retrain with --balance weights.",
        )
    preprocess = preprocess_step(pipeline)
    prep: ColumnPrep = preprocess.named_steps["prep"]
    notes = []
    threshold = meta.get("threshold")
    if threshold is not None:
        notes.append(
            f"This model uses a tuned decision threshold of {threshold:.3f}: predict the positive class when its probability ≥ {threshold:.3f} (the ONNX label output uses 0.5)."
        )
    if meta.get("log_target"):
        notes.append("This model predicts log(1 + target): apply exp(x) - 1 to the ONNX output.")

    _register_missing_indicator()
    classifier = task == CLASSIFICATION
    types = [(c, FloatTensorType([None, 1])) for c in prep.numeric_out] + [
        (c, StringTensorType([None, 1])) for c in prep.categorical
    ]
    graph = Pipeline(
        [("encode", _onnx_friendly(preprocess.named_steps["encode"])), ("model", estimator)]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = skl2onnx.convert_sklearn(
                graph,
                initial_types=types,
                options={id(estimator): {"zipmap": False}}
                if classifier and hasattr(estimator, "predict_proba")
                else None,
                target_opset=TARGET_OPSET,
            )
    except Exception as exc:
        raise PlainMLError(
            f"Couldn't convert this model to ONNX: {str(exc).splitlines()[0][:160]}",
            hint="Models made only of scikit-learn parts convert best: try --models rf,extratrees,linear,tree",
        ) from exc

    columns = [c for c, _ in types]
    names = {
        column: graph_input.name
        for column, graph_input in zip(columns, onx.graph.input, strict=False)
    }
    path = Path(output) if output else Path(model_path).with_suffix(".onnx")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())

    check = _verify(
        path, fitted, prep, names, Path(model_path).parent / HOLDOUT_FILE, meta, classifier
    )
    sidecar = {
        "model": meta.get("model"),
        "task": task,
        "target": meta.get("targets"),
        "classes": meta.get("classes"),
        "inputs": [
            {
                "name": names[c],
                "type": "float32",
                "shape": [None, 1],
                "column": c,
                "from": prep.origin(c),
            }
            for c in prep.numeric_out
        ]
        + [
            {
                "name": names[c],
                "type": "string",
                "shape": [None, 1],
                "column": c,
                "from": c,
                "blank": "",
            }
            for c in prep.categorical
        ],
        "outputs": ["label", "probabilities"] if classifier else ["variable"],
        "notes": [
            'Send each input as a column of shape [rows, 1]. Use NaN for a blank number and "" for a blank category.',
            *(
                [
                    "Date columns are split into parts named COLUMN__year, __month, __day, __dayofweek, __hour, __is_weekend."
                ]
                if prep.datetime
                else []
            ),
            *notes,
        ],
        "verification": check,
    }
    sidecar_path = path.with_name(path.name + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    if verbose:
        with quiet(False):
            success(
                f"Exported {esc(meta.get('model'))} to {esc(path)} ({path.stat().st_size / 1024:.0f} KB)"
            )
            if check.get("rows"):
                if "agreement" in check:
                    info(
                        f"Checked on {check['rows']} rows: ONNX and the original agree on {check['agreement']:.1%} of predictions."
                    )
                elif "probability_difference" in check:
                    info(
                        f"Checked on {check['rows']} rows: probabilities match to within {fmt_num(check['probability_difference'])}."
                    )
                else:
                    info(
                        f"Checked on {check['rows']} rows: typical difference {fmt_num(check['median_difference'])}, "
                        f"largest {fmt_num(check['max_difference'])} ({check['max_relative']:.1%} of the prediction range)."
                    )
                    if check["max_relative"] > 0.001:
                        note(
                            "ONNX runs trees in 32-bit arithmetic, so a few rows can land on the other side of a split."
                        )
            info(f"Input description: {esc(sidecar_path)}")
            for message in notes:
                warn(esc(message))
    return path


def _verify(
    path: Path,
    fitted: Any,
    prep: ColumnPrep,
    names: dict[str, str],
    holdout: Path,
    meta: dict[str, Any],
    classifier: bool,
) -> dict[str, Any]:
    """Compare ONNX and original predictions on the run's test rows."""
    try:
        import onnxruntime as ort
    except ImportError:
        note('Install onnxruntime to double-check the export: pip install "plainml[onnx]"')
        return {}
    if not holdout.is_file():
        return {}
    frame = pd.read_csv(holdout).head(500)
    session = ort.InferenceSession(str(path))
    outputs = session.run(None, onnx_inputs(prep, frame, names))
    expected = fitted.predict(frame)
    if classifier:
        if meta.get("threshold") is not None:
            return {
                "rows": len(frame),
                "probability_difference": float(
                    np.abs(outputs[1] - fitted.predict_proba(frame)).max()
                ),
            }
        agreement = float(
            (np.asarray(outputs[0]).ravel().astype(str) == np.asarray(expected).astype(str)).mean()
        )
        if agreement < 0.99:
            warn(f"ONNX predictions differ from the original on {1 - agreement:.1%} of test rows.")
        return {"rows": len(frame), "agreement": agreement}
    predicted = np.asarray(outputs[0], dtype=float).ravel()
    if meta.get("log_target"):
        predicted = np.expm1(predicted)
    expected = np.asarray(expected, dtype=float)
    differences = np.abs(predicted - expected)
    spread = float(np.ptp(expected)) or 1.0
    return {
        "rows": len(frame),
        "max_difference": float(differences.max()),
        "median_difference": float(np.median(differences)),
        "max_relative": float(differences.max() / spread),
    }
