FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

USER root
# Security: Update and upgrade system packages to patch known vulnerabilities
RUN apt update && apt upgrade -y && apt clean && rm -rf /var/lib/apt/lists/*

USER 1000

###### The below is a workaround for the issue described in the following link:
###### https://github.com/yaml/pyyaml/issues/601#issuecomment-1813963845
RUN pip3 install "cython<3.0.0" wheel && pip3 install "pyyaml==5.4.1" --no-build-isolation
###### 
RUN pip3 install --user transformers>=4.30.2 git+https://github.com/mlfoundations/open_lm.git

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.4 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/huggingface-adapter:0.1.4

# run tail -f /dev/null to keep the container running
CMD ["tail", "-f", "/dev/null"]