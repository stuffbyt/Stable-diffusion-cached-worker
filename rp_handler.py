import os
import base64
from io import BytesIO
from pathlib import Path

import torch
import runpod
from diffusers import DiffusionPipeline

# RunPod Cached Models live here (HF cache layout)
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

# Force offline so we only ever load from the cached snapshot
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def normalize_model_id(raw: str) -> str:
    """
    Accepts:
      - "ofa-sys/small-stable-diffusion-v0"
      - "https://huggingface.co/ofa-sys/small-stable-diffusion-v0"
      - "...:revision" (suffix will be ignored)
    Returns:
      - "ofa-sys/small-stable-diffusion-v0" (org lowercased)
    """
    s = (raw or "").strip()
    if not s:
        return "ofa-sys/small-stable-diffusion-v0"

    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1]

    # strip optional ":revision" suffix
    s = s.split(":", 1)[0]

    org, name = s.split("/", 1)
    return f"{org.lower()}/{name}"

MODEL_ID = normalize_model_id(os.getenv("MODEL_ID", "ofa-sys/small-stable-diffusion-v0"))
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "true").lower() == "true"

def model_cache_dir(model_id: str) -> Path:
    org, name = model_id.split("/", 1)
    return CACHE_ROOT / f"models--{org}--{name}"

def resolve_snapshot_path(model_id: str) -> Path:
    mdir = model_cache_dir(model_id)
    if not mdir.exists():
        # helpful debug: show what actually exists in hub
        existing = []
        if CACHE_ROOT.exists():
            try:
                existing = sorted([p.name for p in CACHE_ROOT.iterdir()])[:30]
            except Exception:
                existing = []
        raise FileNotFoundError(
            f"Model cache dir not found: {mdir}. "
            f"Make sure endpoint Cached Model is set to https://huggingface.co/{model_id}. "
            f"Hub entries (first 30): {existing}"
        )

    # Prefer refs/main when present
    ref_main = mdir / "refs" / "main"
    if ref_main.exists():
        rev = ref_main.read_text().strip()
        snap = mdir / "snapshots" / rev
        if snap.exists():
            return snap

    # Fallback: first snapshot directory
    snap_root = mdir / "snapshots"
    snaps = sorted(snap_root.glob("*")) if snap_root.exists() else []
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under {snap_root}")
    return snaps[0]

_PIPE = None

def load_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Ensure the Serverless endpoint is using a GPU.")

    snapshot_path = resolve_snapshot_path(MODEL_ID)

    _PIPE = DiffusionPipeline.from_pretrained(
        str(snapshot_path),
        torch_dtype=torch.float16,
        local_files_only=LOCAL_FILES_ONLY,
    ).to("cuda")

    return _PIPE

def pil_to_b64_png(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt", "a cute corgi astronaut, cinematic lighting")
    steps = int(inp.get("steps", 20))

    pipe = load_pipe()
    image = pipe(prompt=prompt, num_inference_steps=steps).images[0]

    return {
        "model_id": MODEL_ID,
        "prompt": prompt,
        "steps": steps,
        "image_base64_png": pil_to_b64_png(image),
        "cache_root": str(CACHE_ROOT),
        "snapshot_path": str(resolve_snapshot_path(MODEL_ID)),
    }

runpod.serverless.start({"handler": handler})
