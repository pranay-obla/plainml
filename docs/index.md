# plainml

**From a spreadsheet to a trained, explained, deployable machine learning model in one command.**

```bash
pip install plainml
plainml train examples/churn.csv --target churned
```

plainml checks your data, picks classification or regression, compares up to 17 models fairly
(30 with `--thorough`; cross-validation, a held-out test set, a do-nothing baseline), explains the winner
in plain English, and saves it with an HTML report and a model card. It also clusters, finds anomalies,
forecasts (many series, planned inputs, holidays), ranks columns with 19 importance methods, watches for
drift, tunes, serves models as an API, packages them for Docker and exports them to ONNX or MLflow.

Prefer clicking? **[Try plainml in your browser](https://plainml-tpua.vercel.app)**, with nothing to install, or run `plainml web`
for the same website on your own machine.

- New here? Start with the [README quickstart](https://github.com/pranay-obla/plainml#30-second-quickstart),
  run `plainml` for guided mode, or `plainml web` for the website.
- [The website](website.md): upload, run and download in the browser.
- [Command reference](commands.md): every command and option.
- [Python API](python-api.md): use plainml from scripts and notebooks.
- [How it works](how-it-works.md): what happens to your data, and why the scores can be trusted.
- [Hosting](hosting.md): the docs on Vercel, the website on Hugging Face Spaces or another container host.
- [FAQ](faq.md)
