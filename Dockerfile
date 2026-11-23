FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

USER root
# Security: Update and upgrade system packages to patch known vulnerabilities
RUN apt update && apt upgrade -y && apt install -y ffmpeg rustc cargo && apt clean && rm -rf /var/lib/apt/lists/*
# Fix: the dtlpy runner agent hardcodes /usr/bin/python3 which resolves to system Python 3.8.
# Point it to Python 3.10 (the image's intended interpreter) so dtlpy>=1.x installs correctly.
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

USER 1000
COPY requirements.txt /tmp/
WORKDIR /tmp

RUN pip3 install --user -r requirements.txt

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.5 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.5