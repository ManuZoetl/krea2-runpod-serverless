from huggingface_hub import snapshot_download
import os

# Gewichte in einen festen Ordner im Docker-Image sichern
model_dir = "/models/krea2"
os.makedirs(model_dir, exist_ok=True)

# Lade Krea 2 herunter
snapshot_download(repo_id="krea/Krea-2-Turbo", local_dir=model_dir)
