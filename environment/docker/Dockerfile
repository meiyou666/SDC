FROM python:3.11.17-bookworm@sha256:5887f265d8d44d8b4734d3658b21d668edf5aaf92e4945bc0c984afbfab105d3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    MPLBACKEND=Agg

COPY environment/dependencies/requirements.lock /opt/sdc/environment/dependencies/requirements.lock
RUN /usr/local/bin/python -m venv --without-pip /opt/venv \
    && /usr/local/bin/python -m pip --python /opt/venv/bin/python install \
       --require-hashes --only-binary=:all: --no-deps \
       -r /opt/sdc/environment/dependencies/requirements.lock \
    && /opt/venv/bin/python -m pip check

COPY environment/baseline.json environment/verify_environment.py /opt/sdc/environment/
COPY environment/docker/entrypoint.sh /opt/sdc/entrypoint.sh
RUN chmod 755 /opt/sdc/entrypoint.sh \
    && groupadd --gid 1000 sdc \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash sdc \
    && mkdir -p /workspace/SDC \
    && chown sdc:sdc /workspace/SDC

USER sdc
WORKDIR /workspace/SDC
RUN python /opt/sdc/environment/verify_environment.py --smoke
ENTRYPOINT ["/opt/sdc/entrypoint.sh"]
CMD ["bash"]
