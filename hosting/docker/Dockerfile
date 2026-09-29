# plainml's website in a container, for Render, Railway, Fly.io or any Docker host.
# Build it with: docker build -t plainml-web hosting/docker   Full steps: docs/hosting.md.
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

# Run as a normal user (uid 1000, which Hugging Face Docker Spaces also require)
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user/app

EXPOSE 7860

# Runs are kept in /data when a persistent disk is mounted there; otherwise in the app folder,
# which is wiped when the container restarts. Set PLAINML_WEB_TOKEN (a secret environment
# variable) to require an access token.
CMD ["sh", "-c", "runs=/home/user/app/runs; if [ -d /data ] && [ -w /data ]; then runs=/data/runs; fi; exec plainml web --host 0.0.0.0 --port 7860 --no-browser --runs-dir \"$runs\" --max-upload-mb 200"]
