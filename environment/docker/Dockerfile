FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55 AS tooling

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin
RUN sed -i \
       -e 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
       -e 's|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
       /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl git xz-utils libstdc++6 \
    && sed -i 's|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources \
    && rm -rf /var/lib/apt/lists/* \
    && uv python install --default --no-cache 3.11.17

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    XDG_CACHE_HOME=/tmp/sdc-cache \
    PATH=/opt/venv/bin:$PATH \
    MPLBACKEND=Agg

RUN if id ubuntu >/dev/null 2>&1; then \
        usermod --login sdc --home /home/sdc --move-home --shell /bin/bash ubuntu \
        && groupmod --new-name sdc ubuntu; \
    else \
        groupadd --gid 1000 sdc \
        && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash sdc; \
    fi \
    && mkdir -p /workspace/SDC \
    && chown sdc:sdc /workspace/SDC

USER sdc
WORKDIR /workspace/SDC
CMD ["bash"]

FROM tooling AS development
ENV VIRTUAL_ENV=/opt/venv \
    NPM_CONFIG_REGISTRY=https://registry.npmmirror.com \
    NPM_CONFIG_PREFIX=/home/sdc/.local \
    NPM_CONFIG_CACHE=/home/sdc/.cache/npm \
    HF_HOME=/home/sdc/.cache/huggingface \
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
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       openssh-client rsync git git-lfs build-essential cmake ninja-build pkg-config \
       gdb strace tmux htop procps psmisc lsof time \
       iproute2 iputils-ping netcat-openbsd dnsutils \
       ca-certificates curl wget aria2 jq ripgrep less file unzip zip xz-utils zstd \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -m 0700 -o sdc -g sdc /home/sdc/.ssh \
    && mkdir -p /home/sdc/.cache \
    && chown -R sdc:sdc /home/sdc/.cache
WORKDIR /opt/sdc/project
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --all-groups --no-build \
    && touch /opt/venv/.lock \
    && chmod a+rw /opt/venv/.lock
COPY environment/baseline.json environment/verify_environment.py ./environment/
COPY environment/docker/entrypoint.sh /opt/sdc/entrypoint.sh
COPY environment/docker/setup-codex.sh /opt/sdc/setup-codex.sh
COPY environment/docker/codex-default.toml /opt/sdc/codex-default.toml
RUN chmod 755 /opt/sdc/entrypoint.sh
USER sdc
RUN python environment/verify_environment.py --smoke
WORKDIR /workspace/SDC
ENTRYPOINT ["/opt/sdc/entrypoint.sh"]
CMD ["bash"]
