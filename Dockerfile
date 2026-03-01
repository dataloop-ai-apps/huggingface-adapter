FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

USER root
# Security: Update and upgrade system packages to patch known vulnerabilities
RUN apt update && apt upgrade -y && apt install -y ffmpeg rustc cargo && apt clean && rm -rf /var/lib/apt/lists/*

USER 1000
COPY requirements.txt /tmp/
WORKDIR /tmp

RUN pip3 install --user -r requirements.txt

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.8 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.8