# Torch >= 2.6 to satisfy Transformers' torch.load safety check (CVE-2025-32434)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /

# Install Python deps
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY rp_handler.py /rp_handler.py

# Start RunPod serverless handler
CMD ["python", "-u", "/rp_handler.py"]

