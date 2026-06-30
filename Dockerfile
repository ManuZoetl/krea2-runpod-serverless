FROM runpod/base:0.4.0-cuda11.8.0

RUN apt-get update && apt-get install -y git python3-pip

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install runpod diffusers transformers torch accelerate huggingface_hub

# Fehlerbehebung: Wir KOPIEREN das lokal heruntergeladene Modell einfach in den Container
COPY models/krea2 /models/krea2

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]
