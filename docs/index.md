# plainml

**From a spreadsheet to a trained, explained, deployable machine learning model in one command.**

```bash
pip install plainml
plainml train examples/churn.csv --target churned
```

plainml checks your data, picks classification or regression, compares up to 15 models fairly
(cross-validation, a held-out test set, a do-nothing baseline), explains the winner in plain English,
and saves it with an HTML report. It also clusters, finds anomalies, forecasts, tunes, serves models as
an API and exports them to ONNX.

- New here? Start with the [README quickstart](https://github.com/pranay-obla/plainml#30-second-quickstart),
  or run `plainml` for guided mode.
- [Command reference](commands.md): every command and option.
- [Python API](python-api.md): use plainml from scripts and notebooks.
- [How it works](how-it-works.md): what happens to your data, and why the scores can be trusted.
- [FAQ](faq.md)
