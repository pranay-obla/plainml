# Releasing plainml

Releases go to [PyPI](https://pypi.org/project/plainml/) through GitHub Actions
(`.github/workflows/python-publish.yml`), using PyPI's Trusted Publishing: no API token is stored
anywhere.

## One-time setup

1. **PyPI account.** Sign up at [pypi.org](https://pypi.org), verify your email and turn on two-factor
   authentication (PyPI requires it).
2. **Trusted publisher.** On PyPI, go to **Your account → Publishing → Add a new pending publisher**. Use
   *pending* before the first release, since the project doesn't exist yet. Enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `plainml` |
   | Owner | `pranay-obla` |
   | Repository name | `plainml` |
   | Workflow name | `python-publish.yml` |
   | Environment name | `pypi` |

   After the first release it becomes a normal publisher under the project's **Publishing** settings.
3. **Optional approval step.** On GitHub, open **Settings → Environments → pypi** (created on the first
   run, or create it yourself) and add yourself as a required reviewer. Every publish then waits for your
   click.

## Each release

1. Update the version in `plainml/__init__.py` (`__version__ = "0.1.1"`). The package reads its version
   from there.
2. In `CHANGELOG.md`, give the release a date and list what changed.
3. Check everything locally:

   ```bash
   pytest -q
   ```

   ```bash
   python docs/generate_commands.py --check
   ```

4. Commit and push to `main`, and wait for CI to pass.
5. On GitHub, go to **Releases → Draft a new release**, create the tag `v0.1.1` (it must match
   `__version__`), paste the changelog entry and **Publish release**.
6. The *Upload Python Package* workflow then:
   - runs the tests on Python 3.10 and 3.14
   - checks the tag matches the version
   - builds the wheel and source archive, checks them, and publishes them

   Follow it under the **Actions** tab.
7. Check the result in a fresh environment:

   ```bash
   python -m venv /tmp/plainml-check
   ```

   ```bash
   /tmp/plainml-check/bin/pip install "plainml[web]==0.1.1"
   ```

   ```bash
   /tmp/plainml-check/bin/plainml --version
   ```

8. The Vercel site rebuilds from `main` by itself. If you host the website elsewhere, update it: re-run
   `plainml web --export` and re-upload (Hugging Face Static Space), or bump `PLAINML_VERSION` in
   `hosting/docker/Dockerfile` (server version).

PyPI never accepts the same version twice. If something is wrong with a release, fix it and publish the
next version (e.g. `0.1.2`); a broken release can be *yanked* on PyPI so installs skip it.

## Publishing by hand (fallback)

If Actions is unavailable, create an API token on PyPI (**Account settings → API tokens**, scoped to the
`plainml` project), then:

```bash
python -m pip install --upgrade build twine
```

```bash
rm -rf dist && python -m build
```

```bash
twine check dist/*
```

```bash
twine upload dist/*
```

Use `__token__` as the username and the `pypi-…` token as the password.

TestPyPI isn't an option for practice runs: the name `plainml` is already taken there by an unrelated
project. Test the built wheel in a fresh virtual environment instead.
