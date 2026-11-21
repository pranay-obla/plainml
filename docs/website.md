# The website

**Try it now at [plainml-tpua.vercel.app](https://plainml-tpua.vercel.app)**, with nothing to install. That's the in-browser
version described below: everything runs on your own computer.

To run the website on your own machine:

```bash
pip install "plainml[web]"
plainml web
```

A browser tab opens at `http://localhost:8765`. Everything runs on your machine, and results are saved in
the same `runs/` folder the command line uses, so both see the same history.

## Using it

1. **Upload.** Drag a CSV, Excel, Parquet, JSON or TSV file onto the page. You see every column's type
   (number, category, date, text, or not used) and a preview of the first rows.

    No data to hand? Next to the drop zone are six made-up example datasets (churn, house prices, store
    sales, daily sales, customers and card payments). One click loads one, picks the task that suits it
    and fills in its settings, so you only have to press **Run**.
2. **Choose a task.**

    | Task | What you get |
    |---|---|
    | Predict a column | Models compared and the best one explained, as with `plainml train` |
    | Forecast over time | A forecast with an 80% range, optionally one per store/product, with inputs and holidays |
    | Find groups | Segments and what sets each apart |
    | Find unusual rows | Every row scored, the unusual ones flagged with reasons |
    | Rank the columns | Which columns matter, by several importance methods and a consensus |
    | Check for drift | Whether this data differs from a model's training data or another file |
    | Profile the data | Every column summarised, with warnings |
    | Clean the data | A tidied copy to download |

    Each task has its main options up front and the rest under **More options**.
3. **Watch it run.** A progress bar and a live log show each step. Jobs run one at a time, in order.
4. **Read the results.** Headline numbers, the key findings in plain English, a 3D view you can turn
   and zoom, and the full report. The 3D view depends on the task:

    | Task | 3D view |
    |---|---|
    | Predict a column | Every model's cross-validated score as a bar, the winner highlighted, the do-nothing baseline as a line |
    | Rank the columns | Each column's consensus importance; the ones worth keeping highlighted |
    | Find groups | Every row squeezed onto three axes, colored by group |
    | Find unusual rows | The same, with the flagged rows enlarged and highlighted |
    | Forecast over time | History and forecast with its likely range, one lane per store or product |
    | Check for drift | Each column's change (PSI) against the moderate and major thresholds |

    Hover over a bar or point for its details, or switch to **Table** for the same numbers as text.
5. **Download what you need.** Every file the run wrote is listed with what it is and its size, each
   with its own **Download** button. Tables can be previewed first.

**Runs** lists everything you've run (from the website or the command line), with search and filters.
**Predict** uses any saved model on a new upload, or continues a forecast.

## Options

```text
plainml web --port 9000                   # another port
plainml web --runs-dir projects/churn     # keep runs somewhere else
plainml web --max-upload-mb 2000          # allow bigger uploads (default 500 MB)
plainml web --no-browser                  # don't open a tab
```

## Sharing it on a network

By default only your own machine can open the site. To let colleagues use it:

```bash
plainml web --host 0.0.0.0 --token SOMETHING-LONG-AND-RANDOM
```

Visitors enter the token once (their browser remembers it for 30 days). Anyone with it can upload data and run jobs, so share it only on a
network you trust, or put the site behind HTTPS (for example a reverse proxy). The token can also come from
the `PLAINML_WEB_TOKEN` environment variable, which keeps it out of your shell history.

## In the browser, with no server

```bash
plainml web --export site
```

This writes the same website as static files. Opened in a browser, it starts Python (Pyodide) in the
background, installs plainml, and runs every task on the visitor's own computer, so their data is never
uploaded:
- The first visit downloads about 50 MB of Python libraries; after that they're cached.
- Runs are saved in that browser.
- Downloads are saved straight from the page.
- Uploads are limited to 200 MB, and training uses one CPU core, so very large jobs are better with the
  command line or the server version.

Host the folder at the root of a domain on anything that serves static files (its pages refer to
`/_next/...` and `/worker.js`). The repository's `vercel.json` publishes it on
Vercel together with these docs. See [Hosting](hosting.md) for the steps, including a free Hugging Face
Static Space.

## Putting the server version online

The server version needs a host that runs an always-on container, such as Render, Railway or Fly.io.
The repository includes a Dockerfile for it. See [Hosting](hosting.md).

## Working on the website

The page is a Next.js (React) app in the repository's `frontend/` folder. `npm run build` writes static
files and copies them into `plainml/web/static/`, which is what `plainml web` serves and what ships in
the package. A source checkout needs that build once (Node.js 20.9 or newer):

```bash
cd frontend && npm ci && npm run build
```

For live editing, start plainml and the dev server side by side; the dev server forwards `/api` to
plainml (set `PLAINML_API` if it isn't on port 8765):

```bash
plainml web --no-browser
```

```bash
cd frontend && npm run dev
```

## Embedding it

The site is a FastAPI app, so you can mount it inside another one:

```python
from plainml.web.server import create_app

app = create_app("runs", token="change-me")
```
