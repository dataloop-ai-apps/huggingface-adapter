FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.devel.py3.8.pytorch2

USER root
RUN apt update && apt install -y ffmpeg

USER 1000
COPY requirements.txt /tmp/
WORKDIR /tmp

RUN pip3 install --user -r requirements.txt

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.4 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.4