FROM runpod/base:0.4.0-cuda11.8.0

RUN apt-get update && apt-get install -y git python3-pip

RUN pip3 install runpod diffusers transformers torch accelerate huggingface_hub

COPY download_models.py .

# Nutzt das GitHub-Secret sicher während des Build-Vorgangs, ohne es im Image zu speichern
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python3 download_models.py

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]
