# plainml's website as a Hugging Face Space (Docker SDK).
# Put this file and the README.md next to it into a new Docker Space. Full steps: docs/hosting.md.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_CACHE_DIR=1

# LightGBM and XGBoost need the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The plainml release to run. Change it (and rebuild the Space) to upgrade.
ARG PLAINML_VERSION=0.1.0
RUN pip install "plainml[web,boost,forecast]==${PLAINML_VERSION}"
# Before plainml is on PyPI, install straight from GitHub instead:
# RUN pip install "plainml[web,boost,forecast] @ https://github.com/pranay-obla/plainml/archive/refs/heads/main.zip"

# Spaces run the container as user 1000
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user/app

EXPOSE 7860

# Runs are kept in /data when the Space has persistent storage; otherwise in the app folder,
# which is wiped when the Space restarts. Set PLAINML_WEB_TOKEN as a Space secret to require
# an access token.
CMD ["sh", "-c", "runs=/home/user/app/runs; if [ -d /data ] && [ -w /data ]; then runs=/data/runs; fi; exec plainml web --host 0.0.0.0 --port 7860 --no-browser --runs-dir \"$runs\" --max-upload-mb 200"]
