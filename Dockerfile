FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- venv 1: PersonaPlex (torch<2.5) -----------------------------------
RUN python3 -m venv /opt/venv-personaplex
RUN git clone --depth 1 https://github.com/NVIDIA/personaplex.git /opt/personaplex-code
RUN /opt/venv-personaplex/bin/pip install --no-cache-dir \
    torch==2.4.1 torchaudio -e /opt/personaplex-code/moshi 'sphn==0.1.12' huggingface_hub
# Real upstream bug fix (see BUILD_STATUS.md Step 2 & 11): request.query["seed"]
# and opus_reader.read_pcm() returning None both crash the server unpatched.
RUN sed -i 's/seed = int(request\["seed"\]) if "seed" in request.query else None/seed = int(request.query["seed"]) if "seed" in request.query else None/' \
    /opt/personaplex-code/moshi/moshi/server.py \
    && sed -i 's/pcm = opus_reader.read_pcm()/pcm = opus_reader.read_pcm()\n                if pcm is None:\n                    continue/' \
    /opt/personaplex-code/moshi/moshi/server.py

# --- venv 2: AVTR-1 (torch>=2.5,<2.8 + TensorRT) ------------------------
RUN python3 -m venv /opt/venv-avtr1
RUN git clone --depth 1 https://github.com/avaturn-live/avtr-1.git /opt/avtr1-code
RUN /opt/venv-avtr1/bin/pip install --no-cache-dir \
    "torch>=2.5.1,<2.8" --index-url https://download.pytorch.org/whl/cu128
RUN /opt/venv-avtr1/bin/pip install --no-cache-dir \
    "tensorrt<=10.12" "onnxruntime-gpu==1.20.1" huggingface_hub imageio imageio-ffmpeg \
    opencv-python-headless tqdm attrs tenacity kornia scikit-image soxr roma av soundfile \
    safetensors httpx fastapi uvicorn python-multipart onnx onnx-graphsurgeon anyio einops \
    numpy websockets Pillow -e /opt/avtr1-code

# --- main app venv (lightweight, no torch) ------------------------------
RUN python3 -m venv /opt/venv-app
RUN /opt/venv-app/bin/pip install --no-cache-dir \
    fastapi uvicorn websockets soxr sphn numpy requests python-multipart

COPY . /app
RUN chmod +x /app/entrypoint.sh
ENV PYTHONPATH=/app
# onnxruntime's CUDA execution provider needs cuDNN discoverable (see
# BUILD_STATUS.md Step 8 -- pip-installed nvidia/*/lib dirs aren't on the
# loader path by default).
ENV LD_LIBRARY_PATH=/opt/venv-avtr1/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv-avtr1/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv-avtr1/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv-avtr1/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venv-avtr1/lib/python3.12/site-packages/nvidia/curand/lib:/opt/venv-avtr1/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/venv-avtr1/lib/python3.12/site-packages/nvidia/cusparse/lib

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
