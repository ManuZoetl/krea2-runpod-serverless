import runpod
import torch
import io
from diffusers import DiffusionPipeline

print("⚡ Kaltstart: Lade Krea 2 von der lokalen SSD in den VRAM...")

# Das Modell wird SOFORT geladen, wenn der Container anspringt
# Da die Gewichte im Docker-Image (/models/krea2) liegen, geht das extrem schnell
pipe = DiffusionPipeline.from_pretrained(
    "/models/krea2",
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("✅ Modell ist einsatzbereit!")

# Die Core-Funktion, die bei jedem API-Aufruf getriggert wird (Warm-Start)
def handler(job):
    # Parameter aus dem API-Request ziehen
    job_input = job['input']
    prompt = job_input.get("prompt", "A beautiful digital artwork")
    
    # Krea 2 Turbo Inferenz (8 Steps für maximale Geschwindigkeit)
    image = pipe(
        prompt=prompt,
        num_inference_steps=8,
        guidance_scale=1.5
    ).images[0]
    
    # Bild in Bytes umwandeln (kannst du als Base64 oder direkt zurückgeben)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    # RunPod erwartet ein JSON-kompatibles Wörterbuch als Rückgabe
    import base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    return {"image_base64": encoded_image}

# Handler bei RunPod registrieren
runpod.serverless.start({"handler": handler})
