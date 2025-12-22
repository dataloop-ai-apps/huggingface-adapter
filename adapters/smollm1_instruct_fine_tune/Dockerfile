FROM dataloopai/dtlpy-agent:gpu.cuda.12.4.py3.10.opencv

USER root

RUN apt-get update && apt-get install -y curl

USER 1000
COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
ENV HOME=/tmp

# Install packages
RUN pip install --user -r requirements.txt
