FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

USER root

ENV HF_HOME=/tmp/.cache/huggingface
ENV HF_HUB_CACHE=/tmp/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub

RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision

RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U --no-cache-dir \
  "transformers==4.57.3" \
  "huggingface-hub>=0.34.0,<1.0" \
  safetensors

RUN ${DL_PYTHON_EXECUTABLE} - <<'PY'
import os
from transformers import Owlv2Processor, Owlv2ForObjectDetection

model_id = "google/owlv2-base-patch16"
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
print("Pre-downloading:", model_id)

Owlv2Processor.from_pretrained(model_id)
Owlv2ForObjectDetection.from_pretrained(model_id)

print("Done.")
PY

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/owlv2-adapter:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/owlv2-adapter:0.0.1

