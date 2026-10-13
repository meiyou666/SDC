ARG ASCEND_SOC=910b
FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 AS cuda_sdk
FROM ascendai/cann:9.1.0-910b-ubuntu22.04-py3.11-devel@sha256:7f0dc8f9525b6c21dd554cb95184de94d4bf9531699cc0bee6150fefe5466a16 AS cann_910b
FROM ascendai/cann:9.1.0-a3-ubuntu22.04-py3.11-devel@sha256:690752f297817231736c7cd07c5637b8de6fbb9e42cf9f8f0499d6b32143c623 AS cann_a3
FROM ascendai/cann:9.1.0-950-ubuntu22.04-py3.11-devel@sha256:c8c8b5a800b583ffdf52a5b53e64992b5d327534d56f8eb20bb379b3e0846438 AS cann_950
FROM cann_${ASCEND_SOC} AS cann_sdk
FROM ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401 AS tooling

ARG SDC_APT_MIRROR
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_BIN_DIR=/usr/local/bin
COPY environment/docker/install-system.sh /opt/sdc/install-system.sh
RUN bash /opt/sdc/install-system.sh minimal \
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
RUN groupadd --gid 1000 sdc \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash sdc \
    && mkdir -p /workspace/SDC && chown sdc:sdc /workspace/SDC
USER sdc
WORKDIR /workspace/SDC
CMD ["bash"]

FROM tooling AS common
ENV VIRTUAL_ENV=/opt/venv \
    NPM_CONFIG_REGISTRY=https://registry.npmmirror.com \
    NPM_CONFIG_PREFIX=/home/sdc/.local \
    NPM_CONFIG_CACHE=/home/sdc/.cache/npm \
    HF_HOME=/home/sdc/.cache/huggingface \
    PYTHONPATH=/workspace/SDC/src \
    PATH=/home/sdc/.local/bin:$PATH
USER root
RUN bash /opt/sdc/install-system.sh full
ARG SDC_INSTALL_AI_TOOLS=true
COPY environment/docker/install-node.sh /opt/sdc/install-node.sh
RUN install -d -o sdc -g sdc /home/sdc/.local \
    && if [ "$SDC_INSTALL_AI_TOOLS" = true ]; then \
         bash /opt/sdc/install-node.sh \
         && npm install --global --ignore-scripts --no-audit --no-fund @openai/codex@latest \
         && npm cache clean --force; \
       elif [ "$SDC_INSTALL_AI_TOOLS" != false ]; then \
         echo 'SDC_INSTALL_AI_TOOLS must be true or false' >&2; exit 1; \
       fi \
    && chown -R sdc:sdc /home/sdc/.local \
    && install -d -m 0700 -o sdc -g sdc /home/sdc/.codex /home/sdc/.ssh \
    && install -d -o sdc -g sdc /home/sdc/.cache
WORKDIR /opt/sdc/project
COPY pyproject.toml uv.lock .python-version ./
COPY environment/baseline.json environment/verify_environment.py ./environment/
COPY environment/docker/entrypoint.sh /opt/sdc/entrypoint.sh
COPY environment/docker/setup-codex.sh /opt/sdc/setup-codex.sh
COPY environment/docker/login-codex.py /opt/sdc/login-codex.py
COPY environment/docker/codex-default.toml /opt/sdc/codex-default.toml
COPY environment/docker/install-python.sh /opt/sdc/install-python.sh
COPY environment/docker/activate-accelerator.sh /opt/sdc/activate-accelerator.sh
RUN ln -s /opt/sdc/activate-accelerator.sh /etc/profile.d/sdc-accelerator.sh
RUN chmod 755 /opt/sdc/entrypoint.sh
RUN bash /opt/sdc/install-python.sh common
WORKDIR /workspace/SDC
ENTRYPOINT ["/opt/sdc/entrypoint.sh"]
CMD ["bash"]

FROM common AS development
ENV SDC_RUNTIME=development
RUN bash /opt/sdc/install-python.sh development
USER sdc
RUN python /opt/sdc/project/environment/verify_environment.py --smoke

FROM common AS nvidia
COPY --from=cuda_sdk /usr/local/cuda-12.8 /usr/local/cuda-12.8
ENV SDC_RUNTIME=nvidia CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN bash /opt/sdc/install-python.sh nvidia
USER sdc
RUN python /opt/sdc/project/environment/verify_environment.py --smoke

FROM common AS ascend
ARG ASCEND_SOC
COPY --from=cann_sdk /usr/local/Ascend /usr/local/Ascend
ENV SDC_RUNTIME=ascend SDC_ASCEND_SOC=${ASCEND_SOC}
RUN bash /opt/sdc/install-python.sh ascend
USER sdc
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 python /opt/sdc/project/environment/verify_environment.py --smoke
