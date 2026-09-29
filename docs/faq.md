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
data, for example) are skipped automatically, and the reason is shown. `--thorough` is the slow option: it
adds more models and a stacked ensemble.

**My file is too big.**
Files over 200 MB load faster with the `fast` extra (polars). Train on a sample (`--sample 200000`); a
few hundred thousand rows is usually plenty. Then predict on the whole file in pieces:
`plainml predict latest huge.csv -o predictions.csv --chunk-size 200000`.

**Does it do deep learning?**
Install the `torch` extra and a PyTorch neural network joins the `--thorough` models (or run it alone with
`--models torch`). It trains on the CPU unless you set `PLAINML_TORCH_DEVICE=cuda` or `mps`. On
spreadsheet-style data, gradient boosting usually wins; the network helps most inside ensembles.

**Can I stop training early?**
Press Ctrl-C once: plainml finishes with the models trained so far. Press it again to quit.

**How do I use the model in production?**
- `plainml serve MODEL` gives you a REST API. Add `--api-key` to require a key.
- `plainml deploy MODEL` writes a Docker build folder for that API: pinned library versions, a non-root
  image and a health check. `docker build` it and run it anywhere containers run, with
  `-e PLAINML_API_KEY=...`.
- In Python, `joblib.load("model.joblib").predict(dataframe)` works on raw rows.
- For other languages, `plainml export MODEL` writes ONNX. `--format mlflow` saves it for an MLflow model
  registry.

Only load model files you trust: loading a joblib/pickle file can run code.

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

**Can I put the website online?**
Yes, for free. `plainml web --export site` writes a version that runs plainml inside each visitor's
browser, as plain files. Host them on Vercel (the repository's `vercel.json` does it, docs included),
GitHub Pages or a Hugging Face Static Space. Visitors' data never leaves their computer. For very large
files or long jobs, run the server version on Render, Railway or Fly.io with the included Dockerfile.
See [Hosting](hosting.md).

**Is it reproducible?**
Yes: fixed seeds, and each run saves its settings, library versions and a fingerprint of the data.
`plainml train --config runs/<run>/config.yaml` repeats it, and `plainml compare` warns when two runs
used different data.
