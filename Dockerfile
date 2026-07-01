FROM runpod/base:0.4.0-cuda11.8.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIFFUSERS_OFFLINE=1 \
    RUNPOD_INIT_TIMEOUT=800 \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /requirements.txt

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
