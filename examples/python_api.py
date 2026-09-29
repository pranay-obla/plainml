"""plainml from Python: train, inspect, predict, explain, rank columns, check drift.

Run:  python examples/python_api.py
"""

from pathlib import Path

import pandas as pd

import plainml

HERE = Path(__file__).parent

# 1. Train. Every CLI option is a keyword argument.
result = plainml.train(HERE / "churn.csv", target="churned", quick=True, verbose=False)
print(
    f"Best model: {result.best_model}  ({result.metric} = {result.cv_score:.3f} in cross-validation)"
)
print("On unseen rows:", {k: round(v, 3) for k, v in result.holdout_scores.items()})
print(result.leaderboard[["model", "f1", "accuracy", "roc_auc"]].head())

# 2. What drives it?
explanation = plainml.explain(result.run_dir, verbose=False)
for sentence in explanation.sentences:
    print("•", sentence)

# 3. Predict on new rows (here: a few rows of the same file, without the answer column).
new_customers = pd.read_csv(HERE / "churn.csv").drop(columns="churned").head(5)
predictions = plainml.predict(result.run_dir, new_customers, proba=True)
print(predictions[["customer_id", "predicted_churned", "confidence"]])

# 4. The saved model is a plain scikit-learn object that accepts raw rows.
model = result.model
print(model.predict(new_customers))

# 5. Which columns matter? Several methods, combined into one ranking.
ranked = plainml.feature_importance(
    HERE / "churn.csv", "churned", methods=["mutual_info", "random_forest", "rfe"], verbose=False
)
print("Worth keeping:", ranked.selected)

# 6. Has new data drifted from what the model learned on? (Here: the same file, so no.)
drift = plainml.check_drift(result.run_dir, HERE / "churn.csv", save=False, verbose=False)
print("Drift:", drift.verdict)

print(f"Report: {result.report_path}")
print(f"Model card: {result.run_dir / 'model_card.md'}")
