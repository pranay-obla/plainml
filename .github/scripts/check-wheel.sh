#!/usr/bin/env bash
# Installs the built wheel in a fresh environment and checks the website really works from it:
# the page, the in-browser worker and the API all answer, and the API reports the expected version.
# Usage: .github/scripts/check-wheel.sh [expected-version]   (run after `python -m build`)
set -euo pipefail

wheel=$(ls dist/*.whl)
contents=$(unzip -l "$wheel")  # not piped into grep -q, which would end unzip early under pipefail
grep -q "plainml/web/static/index.html" <<<"$contents" || { echo "The wheel has no website"; exit 1; }
grep -q "plainml/web/static/worker.js" <<<"$contents" || { echo "The wheel has no worker.js"; exit 1; }

env_dir=$(mktemp -d)
python -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --quiet "${wheel}[web]"
expected=${1:-$("$env_dir/bin/python" -c "import plainml; print(plainml.__version__)")}

cd "$env_dir"  # so the source checkout can't be imported by mistake
"$env_dir/bin/plainml" web --port 8765 --no-browser --runs-dir "$env_dir/runs" &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT

for _ in $(seq 60); do curl -fs http://127.0.0.1:8765/api/info > info.json && break; sleep 1; done
"$env_dir/bin/python" - "$expected" <<'PY'
import json, sys
info = json.load(open("info.json"))
assert info["version"] == sys.argv[1], f"the API says {info['version']}, expected {sys.argv[1]}"
assert info["samples"], "no example datasets"
print(f"plainml {info['version']}: API answers, {len(info['samples'])} example datasets")
PY
curl -fs http://127.0.0.1:8765/ | grep -q "<title>plainml" || { echo "The page isn't served"; exit 1; }
for page in new analyze jobs run runs predict; do
  curl -fs "http://127.0.0.1:8765/$page/" | grep -q "<title>plainml" || { echo "/$page/ isn't served"; exit 1; }
done
curl -fs -o /dev/null http://127.0.0.1:8765/worker.js || { echo "worker.js isn't served"; exit 1; }
echo "The website works from the wheel."
