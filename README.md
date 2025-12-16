# Custom Serverless Worker: Cached Models + Diffusers (Text → Image)

This repo contains a **custom RunPod Serverless worker** that generates an **image from a text prompt** using **Diffusers**, and loads the model weights from **RunPod Cached Models** on disk (offline) rather than downloading at runtime.

---

## How It Works

1. In the RunPod endpoint config, you set a **Cached Model** (Hugging Face repo URL).
2. RunPod places the worker on a host that already has the model cached, or downloads it before starting the worker.
3. The worker resolves the cached snapshot path under:
   ```
   /runpod-volume/huggingface-cache/hub/models--ORG--NAME/snapshots/<REV_OR_HASH>/
   ```
4. The worker forces **offline mode** to ensure it only loads from the cached snapshot.

---

## Repo Structure

```
.
├── Dockerfile
├── requirements.txt
└── rp_handler.py
```

---

## Requirements

- Docker
- RunPod account + API key
- A GPU-capable Serverless endpoint
- (Optional) Hugging Face token for gated/private models

---

## Files

### `requirements.txt`

```txt
runpod
diffusers
transformers
accelerate
safetensors
pillow
```

### `Dockerfile` (GPU-capable base image)

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
```

> **Note:** `rp_handler.py` must exist in the same directory as the Dockerfile, or the build will fail.

---

## Build & Push (IMPORTANT: linux/amd64)

### Apple Silicon (recommended)

```bash
docker buildx build --platform linux/amd64 \
  -t <DOCKER_USER>/sd-cached-worker:1.0 \
  --push .
```

### Intel / amd64 host

```bash
docker build --platform linux/amd64 -t <DOCKER_USER>/sd-cached-worker:1.0 .
docker push <DOCKER_USER>/sd-cached-worker:1.0
```

---

## Create the RunPod Serverless Endpoint

1. Create an endpoint using image:
   ```
   <DOCKER_USER>/sd-cached-worker:1.0
   ```

2. Set **Model (optional)** (Cached Models) to a Hugging Face repo URL, for example:
   ```
   https://huggingface.co/OFA-Sys/small-stable-diffusion-v0
   ```

3. Set environment variables:
   ```
   MODEL_ID=OFA-Sys/small-stable-diffusion-v0
   LOCAL_FILES_ONLY=true
   ```

4. If using gated/private models:
   ```
   HUGGINGFACE_HUB_TOKEN=<your_token>
   ```

---

## Run: Text → Image (runsync)

### Send a request

```bash
export RUNPOD_API_KEY="YOUR_API_KEY"
export ENDPOINT_ID="YOUR_ENDPOINT_ID"

curl -sS --request POST \
  --url "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "accept: application/json" \
  -H "authorization: $RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{ "input": { "prompt": "a watercolor painting of a tiger reading a book", "steps": 25 } }' \
  > resp.json
```

### Decode the PNG

```bash
python - <<'PY'
import json, base64

j = json.load(open("resp.json"))
b64 = j["output"]["image_base64_png"]
open("out.png", "wb").write(base64.b64decode(b64))
print("wrote out.png")
PY
```

Open `out.png` to view your generated image.
