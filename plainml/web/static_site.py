"""``plainml web --export``: the website as static files that run plainml in the browser.

The folder needs no server. A Web Worker (``worker.js``) starts Pyodide, installs plainml and
answers the page's requests inside the visitor's browser, so their data never leaves their
computer. Host it anywhere that serves static files: Vercel, GitHub Pages, Netlify, a Hugging
Face Static Space...

From a source checkout the folder includes a plainml wheel built from that source, so the
site matches the code exactly. Otherwise the browser installs this plainml version from PyPI.

Standard library only, so Vercel's build can run it without installing plainml:

    python -m plainml.web.static_site OUT_DIR
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from plainml import __version__
from plainml._wheel import build_wheel, checkout_root

STATIC = Path(__file__).with_name("static")
MODE = '<meta name="plainml-mode" content="browser">'


class ExportError(RuntimeError):
    pass


def export_static(out_dir: str | Path, *, bundle_wheel: bool | None = None) -> Path:
    """Write the in-browser website to ``out_dir``. Returns the folder.

    ``bundle_wheel`` defaults to True when running from a source checkout.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file in STATIC.iterdir():
        if file.is_file():
            shutil.copy2(file, out / file.name)
    index = out / "index.html"
    page = index.read_text(encoding="utf-8")
    if MODE not in page:
        page = page.replace('<meta charset="utf-8">', f'<meta charset="utf-8">\n{MODE}', 1)
    index.write_text(page, encoding="utf-8")

    manifest: dict[str, str | None] = {"version": __version__, "wheel": None}
    source = checkout_root()
    if bundle_wheel is None:
        bundle_wheel = source is not None
    if bundle_wheel:
        if source is None:
            raise ExportError("Bundling a wheel needs a plainml source checkout.")
        wheels = out / "wheels"
        shutil.rmtree(wheels, ignore_errors=True)
        wheel = build_wheel(source, wheels)
        if wheel is None:
            raise ExportError("Couldn't build plainml's wheel (is pip available?).")
        # A folder named after the content, so browsers never reuse a cached older build
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()[:12]
        (wheels / digest).mkdir()
        wheel = wheel.rename(wheels / digest / wheel.name)
        manifest["wheel"] = f"wheels/{digest}/{wheel.name}"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    out = export_static(args[0] if args else "site")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["wheel"] or f"plainml=={manifest['version']} from PyPI"
    print(f"Wrote the in-browser plainml website to {out} (plainml: {source})")


if __name__ == "__main__":
    main()
