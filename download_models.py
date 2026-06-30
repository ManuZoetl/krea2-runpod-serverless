import os
from huggingface_hub import snapshot_download

# Prüfen, ob das Token im Container ankommt
token = os.getenv("HF_TOKEN")
if not token:
    print("❌ FEHLER: Kein HF_TOKEN in den Umgebungsvariablen gefunden!")
else:
    print(f"🔑 HF_TOKEN gefunden (Länge: {len(token)} Zeichen). Starte Download...")

model_dir = "/models/krea2"
os.makedirs(model_dir, exist_ok=True)

try:
    # Lade Krea 2 mit dem übergebenen Token herunter
    snapshot_download(
        repo_id="krea/Krea-2-Turbo", 
        local_dir=model_dir,
        token=token
    )
    print("✅ Download erfolgreich abgeschlossen!")
except Exception as e:
    print(f"❌ Download fehlgeschlagen: {str(e)}")
    raise e
