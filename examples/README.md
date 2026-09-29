# Examples

Six small, synthetic datasets (no real people) to try every part of plainml. They're deliberately a bit
messy, like real exports: blank cells, prices stored as text, day-first dates, ID columns, free-text notes.
Regenerate them with `python examples/make_datasets.py`.

| File | Rows | Try it with |
|---|---|---|
| `churn.csv` | 1,500 | classification: will a subscriber cancel? (imbalanced: 18% yes) |
| `house_prices.csv` | 1,200 | regression: what will a house sell for? |
| `customers.csv` | 600 | clustering: which customer segments exist? |
| `transactions.csv` | 3,000 | anomaly detection: which payments look suspicious? (`is_fraud` marks the truth) |
| `daily_sales.csv` | 912 days | forecasting: how many units will sell next month? |
| `store_sales.csv` | 3 stores × 546 days | forecasting many series, with planned promotions and holidays |

Run these from the repository root.

## Classification: churn

```bash
plainml profile examples/churn.csv --target churned     # look before you leap
plainml train examples/churn.csv --target churned --open
```

Things to notice:

- `customer_id` is set aside automatically (it's an ID), `monthly_charges` ("$59.00") is read as a number,
  `signup_date` (day/month/year) is split into date parts, and `last_feedback` is treated as free text.
- The classes are imbalanced, so plainml weights the rare "yes" class and tunes the decision threshold.
- The report explains the drivers, e.g. more support calls → more likely to churn, monthly contracts churn more.

Then:

```bash
plainml explain latest examples/churn.csv --row 0     # why this customer got their prediction
plainml tune latest --trials 30                       # try to beat the default settings
plainml predict latest examples/churn.csv -o predictions.csv --proba
plainml importance examples/churn.csv -t churned --open  # which columns matter, by 8 methods and a consensus
```

## Regression: house prices

```bash
plainml train examples/house_prices.csv -t price --metric mae
plainml select examples/house_prices.csv -t price -o reduced.csv   # keep only the columns that matter
```

Effects read like *"neighborhood = 'lakeview' gives the highest predicted price"*.

## Clustering: customers

```bash
plainml cluster examples/customers.csv -o segments.csv --open
```

Three segments are hidden in the data: young app users, big-spending store shoppers, and frequent web visitors.
plainml should find them and describe each one. `member_id` is ignored automatically.

## Anomaly detection: transactions

```bash
plainml anomaly examples/transactions.csv --label is_fraud     # score the detectors against the truth
plainml anomaly examples/transactions.csv --drop is_fraud --contamination 1%   # as if you had no labels
```

Each flagged row comes with a reason, e.g. *"amount = 5,120 (38× the usual spread above typical)"*.

## Forecasting: daily sales

```bash
plainml forecast examples/daily_sales.csv -t units_sold --horizon 30 --open
```

The history has an upward trend, busy weekends and a yearly cycle; the report shows the backtests and the
forecast with its 80% range.

## Forecasting many series: store sales

```bash
plainml forecast examples/store_sales.csv -t sales --group store --inputs promo --country US --horizon 14 --open
```

Three stores, each with its own size and growth. Promotions lift sales by about a third, and public
holidays cut them. The last 14 rows of each store have a promotion planned but no sales yet. plainml reads
those rows as plans, so the forecast jumps on the planned promotion days. The report shows the total and a
chart for each store; `forecast.csv` has one row per store and day.

## Everything in the browser

```bash
pip install "plainml[web]"
plainml web
```

Drag any of these files onto the page, pick a task, and download whichever result files you want. Or,
with nothing to install, open [plainml-tpua.vercel.app](https://plainml-tpua.vercel.app) and drag them there.

## Deploy

```bash
plainml train examples/churn.csv -t churned --models rf,linear --no-ensemble
plainml serve latest                          # then open http://localhost:8000/docs
plainml export latest -o churn.onnx           # needs: pip install "plainml[onnx]"
plainml deploy latest                         # a Docker build folder for the API, with a README
```

## From Python

See [python_api.py](python_api.py).
