FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

# Security: Update and upgrade system packages to patch known vulnerabilities
# (Optional: on apt mirror hash mismatch, continue so pip dependency build can run)
RUN (apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*) || true

###### The below is a workaround for the issue described in the following link:
###### https://github.com/yaml/pyyaml/issues/601#issuecomment-1813963845
RUN ${DL_PYTHON_EXECUTABLE} -m pip install "cython<3.0.0" wheel && pip3 install "pyyaml==5.4.1" --no-build-isolation
######
# Pre-install numpy + pyarrow from wheels so open_lm's installer does not build them from source (numpy fails on Python 3.10 C API).
RUN ${DL_PYTHON_EXECUTABLE} -m pip install "numpy>=1.21,<3" "pyarrow"
# --no-build-isolation so pip uses pre-installed numpy/pyarrow instead of building numpy==1.19.4 (incompatible with Python 3.10)
RUN ${DL_PYTHON_EXECUTABLE} -m pip install --no-build-isolation transformers>=4.30.2 git+https://github.com/mlfoundations/open_lm.git

# Patch open_lm: make xformers import conditional so torch/xformers version mismatches at runtime don't crash
RUN find / -path "*/open_lm/*.py" -exec sed -i 's/^import xformers\.ops as xops$/try:\n    import xformers.ops as xops\nexcept (ImportError, Exception):\n    xops = None/' {} +

# Allow runtime pip --upgrade --user to modify dist-info metadata (INSTALLER, etc.) in system site-packages
RUN chmod -R o+w /usr/local/lib/python3.10/dist-packages/

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/gpu/huggingface-adapter-dclm-7b:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/gpu/huggingface-adapter-dclm-7b:0.0.1

# run tail -f /dev/null to keep the container running
CMD ["tail", "-f", "/dev/null"]