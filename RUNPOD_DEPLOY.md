# Krea 2 RunPod Serverless Deployment

This worker is designed for RunPod Serverless with RunPod cached Hugging Face models.

The Docker image intentionally does **not** include the Krea 2 model weights. GitHub Actions only builds a small worker image with code and Python dependencies. The endpoint must use RunPod's **Model** field with `krea/Krea-2-Turbo`, so the model is cached under `/runpod-volume/huggingface-cache/hub/`.

## 1. Build the container image

Push or merge to `main`, or run the GitHub Actions workflow manually:

```text
Actions → Build and Push Krea 2 Worker → Run workflow
```

The image is pushed to:

```text
ghcr.io/manuzoetl/krea2-runpod-serverless:latest
```

and also tagged with the commit SHA.

## 2. Make sure RunPod can pull the image

If the GHCR package is private, either make the package public or configure registry credentials in RunPod.

For a quick first deployment, making the GHCR package public is the easiest option:

```text
GitHub → Packages → krea2-runpod-serverless → Package settings → Change visibility → Public
```

## 3. Create the RunPod Serverless endpoint

In RunPod Console:

```text
Serverless → New Endpoint
```

Recommended settings for the first test:

```text
Endpoint type: Queue-based
Container image: ghcr.io/manuzoetl/krea2-runpod-serverless:latest
Model: krea/Krea-2-Turbo
FlashBoot: Enabled
Active workers: 0
Max workers: 1
GPUs per worker: 1
Idle timeout: 60s to 300s
Execution timeout: 600s
```

If Krea 2 is gated/private on Hugging Face, add a Hugging Face token in the endpoint model settings.

## 4. Test with a health job

Use the RunPod endpoint ID and API key:

```bash
export RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"
export ENDPOINT_ID="YOUR_ENDPOINT_ID"

curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input":{"health_check":true}}' | jq
```

Expected output contains:

```json
{
  "ok": true,
  "model_id": "krea/Krea-2-Turbo",
  "cuda_available": true
}
```

## 5. Generate an image

```bash
curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cinematic portrait of a futuristic assistant orb floating in a dark studio, premium product render, high detail",
      "steps": 8,
      "guidance_scale": 1.5,
      "width": 1024,
      "height": 1024,
      "seed": 77
    }
  }' > response.json

cat response.json | jq '{status, delayTime, executionTime, output: (.output | keys)}'
cat response.json | jq -r '.output.image_base64' | base64 -d > krea2_test.jpg
```

## 6. Cold start tuning

Start with:

```text
Active workers: 0
Idle timeout: 60s to 300s
FlashBoot: Enabled
```

If the user experience must be instant, set:

```text
Active workers: 1
```

That keeps one worker warm, but it costs idle GPU time.

## 7. Useful environment variables

The worker supports these optional environment variables:

```text
MODEL_ID=krea/Krea-2-Turbo
MODEL_PATH=/custom/local/model/path
HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
ALLOW_RUNTIME_DOWNLOAD=0
```

`ALLOW_RUNTIME_DOWNLOAD=1` is only for debugging. Production should use RunPod cached models.
