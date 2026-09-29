# Hosting

Where each part of plainml can live online:

| What | Where | Why |
|---|---|---|
| The website, in the browser | Vercel (free), or any static host | plainml runs in the visitor's browser, so plain files are enough |
| This documentation | Vercel, next to the website | Static pages, rebuilt on every push |
| The Python package | [PyPI](https://pypi.org/project/plainml/) | `pip install plainml`. Release steps are in [RELEASING.md](https://github.com/pranay-obla/plainml/blob/main/RELEASING.md) |
| The website, on a server | Render, Railway, Fly.io | For big files and long jobs: see below |
| A trained model's API | Any container host | `plainml deploy` builds the image |

## The website in the browser

`plainml web --export DIR` writes the website as static files: no server needed. When someone opens it,
a background worker in their browser starts Python (via [Pyodide](https://pyodide.org)), installs
plainml, and does all the work right there:
- **Free to host anywhere:** Vercel, GitHub Pages, Netlify, a Hugging Face Static Space.
- **Private:** visitors' data never leaves their computer.
- **Always on:** it never sleeps, and runs are saved in each visitor's browser between visits.
- **Complete:** every task works, including LightGBM and XGBoost.
- **Trade-offs:**
  - The first visit downloads about 50 MB of Python libraries (cached afterwards).
  - Uploads are limited to 200 MB.
  - Training uses one CPU core, so it's slower than the command line on big data.

From a plainml source checkout, the export includes plainml built from that code. Otherwise the browser
installs the same plainml version from PyPI.

To try it locally:

```bash
plainml web --export site
```

```bash
python -m http.server --directory site
```

Then open http://localhost:8000.

## On Vercel (website and docs)

The repository's `vercel.json` builds both parts: the website at the root of the address and this
documentation under `/docs`.

```json
{
  "framework": null,
  "installCommand": "python3 -m venv .venv && .venv/bin/python -m pip install --quiet \"mkdocs-material>=9.5\" \"mkdocs<2\"",
  "buildCommand": ".venv/bin/python -m plainml.web.static_site site && .venv/bin/python -m mkdocs build -d site/docs",
  "outputDirectory": "site"
}
```

The build tools go into a throwaway virtual environment (`.venv`) because Vercel's own Python refuses
package installs (it's managed by uv, per PEP 668).

1. Sign in at [vercel.com](https://vercel.com) with GitHub.
2. Choose **Add New → Project** and import `pranay-obla/plainml`. The settings come from `vercel.json`,
   so you don't need to change anything.
3. Click **Deploy**. The site appears at a `*.vercel.app` address, with the docs at `/docs`. To use your
   own domain, go to **Settings → Domains**.
4. Every push to `main` redeploys both, and pull requests get their own preview link.
5. Put the addresses in `mkdocs.yml` (`site_url: https://…/docs/`) and in the `[project.urls]` of
   `pyproject.toml` (`Homepage` and `Documentation`).

The documentation can also go to GitHub Pages with `.github/workflows/docs.yml`. That workflow only runs
when started by hand from the **Actions** tab. To use it, turn on **Settings → Pages → Source: GitHub
Actions**.

## On a Hugging Face Static Space

Static Spaces are free on Hugging Face.

1. Export the website:

   ```bash
   plainml web --export space
   ```

2. Copy [`hosting/huggingface/README.md`](https://github.com/pranay-obla/plainml/blob/main/hosting/huggingface/README.md)
   into the `space` folder. Its header tells Hugging Face to serve the folder as a static site.
3. On huggingface.co, choose **New Space**, give it a name, and choose **Static** with a **Blank** template.
4. Upload everything in `space/` to it (**Files → Add file → Upload files**, keeping the `wheels` folder),
   or `git push` it to the Space's repository.

Export again and re-upload whenever you want a newer plainml.

## On a server (Render, Railway, Fly.io)

The server version (`plainml web`) suits big files and long jobs, and keeps runs on the server.
[`hosting/docker/Dockerfile`](https://github.com/pranay-obla/plainml/tree/main/hosting/docker) runs it on
any host that runs containers:
- Tell the host the app listens on **port 7860**.
- Set `PLAINML_WEB_TOKEN` as a secret environment variable to require an access token.
- Attach a **persistent disk at `/data`** so runs survive restarts and redeploys.
- Upgrade by changing `PLAINML_VERSION` in the Dockerfile.

Render has a free plan that sleeps after 15 minutes without visitors and has no disk. That's fine for a
demo, since runs are lost on restart. An always-on instance with a disk costs a few dollars a month.

The server version can't run on Vercel itself: Vercel runs Python only as short request handlers, with
4.5 MB request bodies, at most 5 minutes per request on the free plan, and nothing kept between requests.
That's why Vercel hosts the in-browser version instead.

## A model's prediction API

`plainml deploy MODEL` writes a Docker build folder for one trained model's REST API (see
[Deploying](https://github.com/pranay-obla/plainml#deploying)). Build and push that image to any
container host: Google Cloud Run, AWS App Runner, Azure Container Apps, Fly.io, Render or Kubernetes. Set
`PLAINML_API_KEY` there to require an API key.
