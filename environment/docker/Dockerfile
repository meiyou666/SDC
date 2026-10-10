FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM python:3.11.17-bookworm@sha256:5887f265d8d44d8b4734d3658b21d668edf5aaf92e4945bc0c984afbfab105d3 AS tooling

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    XDG_CACHE_HOME=/tmp/sdc-cache \
    PATH=/opt/venv/bin:$PATH \
    MPLBACKEND=Agg

RUN groupadd --gid 1000 sdc \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash sdc \
    && mkdir -p /workspace/SDC \
    && chown sdc:sdc /workspace/SDC

USER sdc
WORKDIR /workspace/SDC
CMD ["bash"]

FROM tooling AS development
ENV VIRTUAL_ENV=/opt/venv \
    NPM_CONFIG_REGISTRY=https://registry.npmmirror.com \
    NPM_CONFIG_PREFIX=/home/sdc/.local \
    PATH=/home/sdc/.local/bin:$PATH
USER root
RUN curl -fsSL --retry 3 https://registry.npmmirror.com/-/binary/node/v24.21.0/node-v24.21.0-linux-x64.tar.xz -o /tmp/sdc-node.tar.xz \
    && echo 'fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6  /tmp/sdc-node.tar.xz' | sha256sum -c - \
    && tar -xJf /tmp/sdc-node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/sdc-node.tar.xz \
    && npm install --global --ignore-scripts --no-audit --no-fund @openai/codex@latest \
    && npm cache clean --force \
    && chown -R sdc:sdc /home/sdc/.local \
    && install -d -m 0700 -o sdc -g sdc /home/sdc/.codex
WORKDIR /opt/sdc/project
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --all-groups --no-build \
    && touch /opt/venv/.lock \
    && chmod a+rw /opt/venv/.lock
COPY environment/baseline.json environment/verify_environment.py ./environment/
COPY environment/codex/ ./environment/codex/
COPY environment/docker/entrypoint.sh /opt/sdc/entrypoint.sh
RUN chmod 755 /opt/sdc/entrypoint.sh
USER sdc
RUN python environment/codex/configure.py \
    && python environment/verify_environment.py --smoke
WORKDIR /workspace/SDC
ENTRYPOINT ["/opt/sdc/entrypoint.sh"]
CMD ["bash"]
