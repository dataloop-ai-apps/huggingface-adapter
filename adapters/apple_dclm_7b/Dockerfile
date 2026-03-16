FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

# Security: Update and upgrade system packages to patch known vulnerabilities
# (Optional: on apt mirror hash mismatch, continue so pip dependency build can run)
RUN (apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*) || true

###### The below is a workaround for the issue described in the following link:
###### https://github.com/yaml/pyyaml/issues/601#issuecomment-1813963845
RUN ${DL_PYTHON_EXECUTABLE} -m pip install "cython<3.0.0" wheel && pip3 install "pyyaml==5.4.1" --no-build-isolation
######
# Pre-install numpy + pyarrow from wheels so open_lm's installer does not build them from source (numpy fails on Python 3.10 C API).
RUN ${DL_PYTHON_EXECUTABLE} -m pip install --user "numpy>=1.21,<3" "pyarrow"
# --no-build-isolation so pip uses pre-installed numpy/pyarrow instead of building numpy==1.19.4 (incompatible with Python 3.10)
RUN ${DL_PYTHON_EXECUTABLE} -m pip install --user --no-build-isolation transformers>=4.30.2 git+https://github.com/mlfoundations/open_lm.git

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/gpu/huggingface-adapter-dclm-7b:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/gpu/huggingface-adapter-dclm-7b:0.0.1

# run tail -f /dev/null to keep the container running
CMD ["tail", "-f", "/dev/null"]