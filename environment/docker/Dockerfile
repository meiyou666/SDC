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
ENV VIRTUAL_ENV=/opt/venv
USER root
WORKDIR /opt/sdc/project
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --all-groups --no-build \
    && touch /opt/venv/.lock \
    && chmod a+rw /opt/venv/.lock
COPY environment/baseline.json environment/verify_environment.py ./environment/
COPY environment/docker/entrypoint.sh /opt/sdc/entrypoint.sh
RUN chmod 755 /opt/sdc/entrypoint.sh
USER sdc
RUN python environment/verify_environment.py --smoke
WORKDIR /workspace/SDC
ENTRYPOINT ["/opt/sdc/entrypoint.sh"]
CMD ["bash"]
