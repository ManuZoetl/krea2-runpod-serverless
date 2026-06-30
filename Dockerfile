FROM runpod/base:0.4.0-cuda11.8.0

RUN apt-get update && apt-get install -y git python3-pip

# Fehlerbehebung: Nutzung von 'python3 -m pip', um die richtige Umgebung zu treffen
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install runpod diffusers transformers torch accelerate huggingface_hub

COPY download_models.py .

# Akzeptiert das Token als Build-Argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Fehlerbehebung: Expliziter Aufruf über 'python3 -m'
RUN python3 download_models.py

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]
