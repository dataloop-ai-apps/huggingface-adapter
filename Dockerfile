FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

USER root
# Security: Update and upgrade system packages to patch known vulnerabilities
RUN apt update && apt upgrade -y && apt install -y ffmpeg rustc cargo && apt clean && rm -rf /var/lib/apt/lists/*
# Pin torch inductor cache path to a writable location.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache
RUN mkdir -p "$TORCHINDUCTOR_CACHE_DIR" && chmod 777 "$TORCHINDUCTOR_CACHE_DIR"

USER 1000
COPY requirements.txt /tmp/
WORKDIR /tmp

RUN pip3 install --user -r requirements.txt

# docker build --no-cache -t hub.dataloop.ai/dtlpy-runner-images/piper/agent/runner/apps/huggingface-adapter:0.1.6  -f Dockerfile .
# docker push hub.dataloop.ai/dtlpy-runner-images/piper/agent/runner/apps/huggingface-adapter:0.1.6