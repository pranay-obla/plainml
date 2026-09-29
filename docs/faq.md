# FAQ

**Which metric should I use?**
The defaults are sensible: F1 for classification (fair when classes are uneven) and RMSE for regression
(error in your target's units). Use `--metric roc_auc` when you care about ranking (who is most likely to
churn), `--metric recall` when missing a positive is costly (fraud, disease), `--metric mae` when big misses
shouldn't count extra, and `--metric r2` for a unit-free "share explained".

**Why is there a "baseline" model?**
It always predicts the most common class (or the average). If the best model barely beats it, the columns
probably don't contain the information needed, and a fancier model won't help.

**My score is suspiciously good. Why?**
Check the warnings for a **leak**: a column that is calculated from the target, or only known after the
outcome (e.g. `refund_amount` when predicting `will_refund`). Drop it with `--drop COLUMN`. If one column
accounts for most of the importance, that's also a hint.

**What does "big train/test gap" mean?**
The model scores much better on rows it trained on than on new rows: it's memorising. Tree ensembles
often do this and still generalise fine, so judge by the cross-validated and test scores.

**My data has blanks, text, dates, currency symbols…**
That's fine: plainml handles them, and the saved model handles them the same way for new data.
Run `plainml profile DATA` to see how each column was understood, or `plainml clean` for a tidy copy.

**A column was ignored. Why?**
It looked like an ID (a different value in every row) or a constant. Force it in with `--keep COLUMN`.

**It's slow.**
Use `--quick`, `--time-budget 5m`, `--models rf,lightgbm`, or `--sample 50000`. Slow models (SVM on large
data, for example) are skipped automatically, and the reason is shown.

**Can I stop training early?**
Press Ctrl-C once: plainml finishes with the models trained so far. Press it again to quit.

**How do I use the model in production?**
`plainml serve MODEL` gives you a REST API (add `--api-key` to require a key). In Python, `joblib.load("model.joblib").predict(dataframe)`
works on raw rows. For other languages, `plainml export MODEL` writes ONNX. Only load model files you trust:
loading a joblib/pickle file can run code.

**How do I know when to retrain?**
Without labels, run `plainml drift MODEL recent_data.csv`: it shows which columns have shifted from the
training data, and how much the model relies on them (`plainml predict` also warns when new rows look
clearly different). With labels, `plainml evaluate MODEL recent_labelled_data.csv` compares today's scores
with the scores at training time.

**Can I trust the probabilities?**
Check the calibration card in the report: rows given "70%" should be positive about 70% of the time.
If it says poorly calibrated, retrain with `--calibrate`.

**Which feature importance method should I use?**
Don't pick one: `plainml importance` runs several and combines them, and the report shows where they
disagree. Filters (mutual information, F-test) are fast but judge columns one at a time; model-based and
permutation importance see interactions; searches (RFE, Boruta, forward/backward) test subsets directly
but take longer. `--methods all` runs all 19.

**How do I share results without sharing the data?**
Train with `--private` (also on `cluster`, `anomaly` and `importance`). The report, run folder and model then
keep only file names, column names and summary numbers: no example rows, raw values or data paths. Check
the report before sending it: column names and category names can still be revealing.

**Can other people use the website?**
`plainml web` listens only on your machine. To share it on a network, start it with
`plainml web --host 0.0.0.0 --token SOMETHING-LONG` and send people the address and token. Anyone with the
token can upload data and run jobs, so only do this on a network you trust, or behind HTTPS.

**How do I put a model into production?**
`plainml deploy MODEL` writes a Docker build folder (pinned library versions, a non-root image, a health
check). `docker build` it and run it anywhere containers run, with `-e PLAINML_API_KEY=...` to require a key.
`plainml export MODEL --format mlflow` saves it for an MLflow model registry.

**Is it reproducible?**
Yes: fixed seeds, and each run saves its settings, library versions and a fingerprint of the data.
`plainml train --config runs/<run>/config.yaml` repeats it, and `plainml compare` warns when two runs
used different data.
