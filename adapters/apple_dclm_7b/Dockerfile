FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2

USER 1000
COPY requirements.txt /tmp/
WORKDIR /tmp

###### The below is a workaround for the issue described in the following link:
###### https://github.com/yaml/pyyaml/issues/601#issuecomment-1813963845
RUN pip3 install "cython<3.0.0" wheel && pip3 install "pyyaml==5.4.1" --no-build-isolation
###### 
RUN pip3 install --user transformers>=4.30.2 git+https://github.com/mlfoundations/open_lm.git

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.4 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.4

# run tail -f /dev/null to keep the container running
CMD ["tail", "-f", "/dev/null"]