# Hosting

Where each part of plainml can live online:

| What | Where | Why |
|---|---|---|
| The Python package | [PyPI](https://pypi.org/project/plainml/) | `pip install plainml`. The release steps are in [RELEASING.md](https://github.com/pranay-obla/plainml/blob/main/RELEASING.md) |
| This documentation | Vercel (or GitHub Pages) | Static pages, free, redeployed on every push |
| The website (`plainml web`) | Hugging Face Spaces, Render, Railway, Fly.io | Needs an always-on container with room for uploads and long jobs |
| A trained model's API | Any container host | `plainml deploy` builds the image |

## The documentation on Vercel

The repository includes a `vercel.json` that builds this site with MkDocs:

```json
{
  "framework": null,
  "installCommand": "python3 -m pip install \"mkdocs-material>=9.5\" \"mkdocs<2\"",
  "buildCommand": "python3 -m mkdocs build",
  "outputDirectory": "site"
}
```

1. Sign in at [vercel.com](https://vercel.com) with GitHub.
2. Choose **Add New → Project** and import `pranay-obla/plainml`. The settings come from `vercel.json`,
   so you don't need to change anything.
3. Click **Deploy**. The site appears at a `*.vercel.app` address. To use your own domain, go to
   **Settings → Domains**.
4. Every push to `main` redeploys the site, and pull requests get their own preview link.
5. Put the address in `mkdocs.yml` (`site_url: https://…`) and in the `[project.urls]` of
   `pyproject.toml` (`Documentation = "https://…"`).

The repository can also publish the same site to GitHub Pages with `.github/workflows/docs.yml`. It only
runs when started by hand from the **Actions** tab, since the docs live on Vercel. To switch, turn on
**Settings → Pages → Source: GitHub Actions** and add a push trigger to that workflow.

## The website on Hugging Face Spaces

[Hugging Face Spaces](https://huggingface.co/spaces) runs a Docker container for free (2 vCPUs, 16 GB of
memory). That's enough for a public plainml demo. The files are in
[`hosting/huggingface/`](https://github.com/pranay-obla/plainml/tree/main/hosting/huggingface).

1. Publish plainml to PyPI first: the Space runs `pip install plainml`. Before that, switch to the
   commented-out line in the Dockerfile that installs from GitHub.
2. On huggingface.co, choose **New Space**. Name it (e.g. `plainml`), choose the **Docker** SDK with a
   **Blank** template, and keep the free **CPU basic** hardware.
3. Upload `hosting/huggingface/Dockerfile` and `hosting/huggingface/README.md` to the Space: **Files → Add
   file → Upload files**, or `git push` to the Space's repository. The README's header tells Hugging Face
   to use Docker and port 7860.
4. To limit who can use it, go to **Settings → Variables and secrets → New secret** and add
   `PLAINML_WEB_TOKEN` with a long random value. Visitors then enter that token once.
5. The Space builds and starts in a few minutes. Open it at `https://<owner>-<space-name>.hf.space`.
   Signing in and uploading work more reliably there than inside the Hugging Face page.

Things to know:
- **Runs are wiped when the Space restarts**, unless you add persistent storage (Settings → Persistent
  storage). With it, the Dockerfile keeps runs in `/data` automatically.
- **Free Spaces sleep** after a while without visitors and wake up on the next visit.
- **Upgrading:** change `PLAINML_VERSION` in the Dockerfile and the Space rebuilds.

## The website on Render, Railway or Fly.io

The same `hosting/huggingface/Dockerfile` works on any host that runs containers:
- Tell the host the app listens on **port 7860**.
- Set `PLAINML_WEB_TOKEN` as a secret environment variable.
- Attach a **persistent disk at `/data`** so runs survive restarts and redeploys.

Expect to pay a few dollars a month for a small always-on instance with a disk.

## Why not the website on Vercel?

Vercel runs Python apps as short-lived functions, which doesn't fit a tool that trains models:
- Request bodies (and so uploads) are limited to 4.5 MB.
- Each request is stopped after 5 minutes on the free plan (about 13 on paid plans).
- Nothing kept in memory or on disk survives between requests, but plainml's jobs run in the background
  and its runs live in a folder.

Static pages, like this documentation, are what Vercel does best.

## A model's prediction API

`plainml deploy MODEL` writes a Docker build folder for one trained model's REST API (see
[Deploying](https://github.com/pranay-obla/plainml#deploying)). Build and push that image to any
container host: Google Cloud Run, AWS App Runner, Azure Container Apps, Fly.io, Render or Kubernetes. Set
`PLAINML_API_KEY` there to require an API key.
