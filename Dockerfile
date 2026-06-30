FROM runpod/base:0.4.0-cuda11.8.0

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y git python3-pip

# Python Bibliotheken installieren
RUN pip3 install runpod diffusers transformers torch accelerate huggingface_hub

# Script zum Herunterladen der Gewichte in den Container kopieren
COPY download_models.py .
RUN python3 download_models.py

# Handler-Code kopieren
COPY handler.py .

# RunPod verlangt diesen Startbefehl
CMD [ "python3", "-u", "/handler.py" ]
