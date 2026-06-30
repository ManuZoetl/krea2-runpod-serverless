import base64
import glob
import io
import os
import time
from pathlib import Path

import runpod
import torch
from diffusers import DiffusionPipeline


MODEL_ID = os.getenv("MODEL_ID", "krea/Krea-2-Turbo")
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
HF_CACHE_ROOT = Path(os.getenv("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"))
ALLOW_RUNTIME_DOWNLOAD = os.getenv("ALLOW_RUNTIME_DOWNLOAD", "0") == "1"


def resolve_cached_model_path(model_id: str) -> str:
    """Resolve RunPod's Hugging Face cached model directory.

    RunPod cached models are mounted under:
      /runpod-volume/huggingface-cache/hub/models--ORG--NAME/snapshots/HASH/
    """
    if MODEL_PATH:
        path = Path(MODEL_PATH)
        if path.exists():
            return str(path)
        raise RuntimeError(f"MODEL_PATH was set but does not exist: {path}")

    cache_dir = HF_CACHE_ROOT / ("models--" + model_id.replace("/", "--"))
    snapshots = sorted(
        glob.glob(str(cache_dir / "snapshots" / "*")),
        key=lambda p: Path(p).stat().st_mtime,
    )

    if snapshots:
        return snapshots[-1]

    if ALLOW_RUNTIME_DOWNLOAD:
        print(
            "⚠️ Cached model not found. Falling back to runtime Hugging Face download. "
            "Use this only for debugging, not production."
        )
        return model_id

    raise RuntimeError(
        f"Cached model not found for {model_id}. Expected snapshots under: "
        f"{cache_dir}/snapshots/*. Configure the RunPod endpoint Model field to "
        f"'{model_id}' and add a Hugging Face token if the model is gated/private."
    )


print("⚡ Cold start: resolving cached Krea 2 model...")
startup_started = time.time()
resolved_model_path = resolve_cached_model_path(MODEL_ID)
print(f"📦 Loading model from: {resolved_model_path}")

pipe = DiffusionPipeline.from_pretrained(
    resolved_model_path,
    torch_dtype=torch.float16,
    local_files_only=not ALLOW_RUNTIME_DOWNLOAD,
)
pipe.to("cuda")

startup_seconds = round(time.time() - startup_started, 2)
print(f"✅ Krea 2 worker ready in {startup_seconds}s")


def handler(job):
    job_input = job.get("input", {}) or {}

    if job_input.get("health_check"):
        return {
            "ok": True,
            "model_id": MODEL_ID,
            "model_path": resolved_model_path,
            "startup_seconds": startup_seconds,
            "cuda_available": torch.cuda.is_available(),
        }

    prompt = job_input.get("prompt", "A beautiful digital artwork")
    negative_prompt = job_input.get("negative_prompt")
    steps = int(job_input.get("steps", 8))
    guidance_scale = float(job_input.get("guidance_scale", 1.5))
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    seed = job_input.get("seed")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    infer_started = time.time()

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
    }

    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if generator is not None:
        kwargs["generator"] = generator

    image = pipe(**kwargs).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_base64": image_base64,
        "format": "jpeg",
        "width": width,
        "height": height,
        "seed": seed,
        "timing": {
            "startup_seconds": startup_seconds,
            "inference_seconds": round(time.time() - infer_started, 2),
        },
    }


runpod.serverless.start({"handler": handler})
