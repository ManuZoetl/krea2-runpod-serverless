FROM runpod/base:0.4.0-cuda11.8.0

RUN apt-get update && apt-get install -y git python3-pip

RUN pip3 install runpod diffusers transformers torch accelerate huggingface_hub

COPY download_models.py .

# Akzeptiert das Token als Build-Argument während GitHub das Image baut
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Führt das Skript aus
RUN python3 download_models.py

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]
