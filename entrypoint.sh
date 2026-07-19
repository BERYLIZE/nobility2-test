#!/bin/bash
set -e

echo "=== Nobility2 startup ==="

# --- 1. PersonaPlex weights ---------------------------------------------
if [ ! -f /data/personaplex/model.safetensors ]; then
  echo "Downloading PersonaPlex weights (~17GB, first boot only)..."
  mkdir -p /data/personaplex
  /opt/venv-personaplex/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('nvidia/personaplex-7b-v1', local_dir='/data/personaplex', token=os.environ.get('HF_TOKEN'))
"
  mkdir -p /data/personaplex/voices
  tar xzf /data/personaplex/voices.tgz -C /data/personaplex/voices
fi

# --- 2. AVTR-1 artifacts + TensorRT engines ------------------------------
cd /opt/avtr1-code
if [ ! -d /data/avtr1-artifacts/main ]; then
  echo "Downloading AVTR-1 artifacts (~26GB, first boot only)..."
  mkdir -p /data/avtr1-artifacts
  ln -sfn /data/avtr1-artifacts artifacts
  /opt/venv-avtr1/bin/python scripts/download_artifacts.py --workers 8
else
  ln -sfn /data/avtr1-artifacts artifacts
fi

if ! find /data/avtr1-artifacts -iname '*.engine' | grep -q .; then
  echo "Building TensorRT engines (~10 min, first boot only, GPU-specific)..."
  # Build on GPU 1 -- the card AVTR-1 actually runs on (see Step 12 note
  # below); TRT engines are tied to the building GPU's architecture.
  CUDA_VISIBLE_DEVICES=1 /opt/venv-avtr1/bin/python scripts/build_engines.py
fi

# --- 3. Reference portrait: convert once, place where AVTR-1 expects it -
mkdir -p /data/avtr1-artifacts/main/avatars_artifacts/reference_frames
if [ ! -f /data/avtr1-artifacts/main/avatars_artifacts/reference_frames/nobility2.png ]; then
  /opt/venv-avtr1/bin/python -c "
from PIL import Image
Image.open('/app/assets/reference/identity.jpg').convert('RGB').save(
    '/data/avtr1-artifacts/main/avatars_artifacts/reference_frames/nobility2.png')
"
fi
cat > /app/config/reference.json <<EOF
{
  "identity_image_path": "/data/avtr1-artifacts/main/avatars_artifacts/reference_frames/nobility2.png",
  "note": "Config-only swap point for AVTR-1's identity lock."
}
EOF

# --- 4. Start the two model services + main app --------------------------
# GPU split (see BUILD_STATUS.md Step 12): PersonaPlex-7B alone needs ~19GB
# (Moshi's own docs: "24GB, no quantization support" -- confirmed via NVIDIA's
# model card recommending A100/H100). AVTR-1 needs ~3.5GB steady-state but
# spikes during rendering. Co-resident on one 24GB card, the two only have
# soft VRAM-sharing tools available (ORT arena caps, kSameAsRequested) which
# don't give hard isolation -- whichever process allocates first wins, the
# other can starve and OOM mid-render (reproduced repeatedly in testing).
# No real deployment of this model class runs two heavy models co-resident
# on one GPU (NVIDIA's own ACE/digital-human blueprint keeps ASR/LLM/render
# on separate GPUs/nodes) -- so each model gets its own dedicated GPU here
# instead of continuing to fight VRAM contention in software.
echo "Starting PersonaPlex service (GPU 0)..."
cd /data/personaplex
CUDA_VISIBLE_DEVICES=0 /opt/venv-personaplex/bin/python -m moshi.server \
  --host 0.0.0.0 --port 8998 \
  --moshi-weight /data/personaplex/model.safetensors \
  --mimi-weight /data/personaplex/tokenizer-e351c8d8-checkpoint125.safetensors \
  --tokenizer /data/personaplex/tokenizer_spm_32k_3.model \
  --voice-prompt-dir /data/personaplex/voices/voices \
  --device cuda > /var/log/personaplex.log 2>&1 &

echo "Starting AVTR-1 rendering service (GPU 1)..."
cd /opt/avtr1-code
CUDA_VISIBLE_DEVICES=1 NOBILITY2_REFERENCE_CONFIG=/app/config/reference.json \
  /opt/venv-avtr1/bin/python /app/services/avtr1_service/server.py > /var/log/avtr1.log 2>&1 &

echo "Waiting for AVTR-1 to finish loading..."
until grep -q "listening on" /var/log/avtr1.log 2>/dev/null; do
  if ! pgrep -f "avtr1_service/server.py" > /dev/null; then
    echo "AVTR-1 service died during load:"; tail -40 /var/log/avtr1.log; exit 1
  fi
  sleep 2
done
echo "AVTR-1 ready."

echo "Waiting for PersonaPlex to be ready..."
until curl -sf http://localhost:8998/ > /dev/null 2>&1; do
  if ! pgrep -f "moshi.server" > /dev/null; then
    echo "PersonaPlex service died during load:"; tail -40 /var/log/personaplex.log; exit 1
  fi
  sleep 2
done
echo "PersonaPlex ready."

start_app() {
  cd /app
  PYTHONPATH=/app /opt/venv-app/bin/python -m uvicorn app:app \
    --host 0.0.0.0 --port 7860 > /var/log/app.log 2>&1 &
}

start_personaplex() {
  cd /data/personaplex
  CUDA_VISIBLE_DEVICES=0 /opt/venv-personaplex/bin/python -m moshi.server \
    --host 0.0.0.0 --port 8998 \
    --moshi-weight /data/personaplex/model.safetensors \
    --mimi-weight /data/personaplex/tokenizer-e351c8d8-checkpoint125.safetensors \
    --tokenizer /data/personaplex/tokenizer_spm_32k_3.model \
    --voice-prompt-dir /data/personaplex/voices/voices \
    --device cuda >> /var/log/personaplex.log 2>&1 &
}

start_avtr1() {
  cd /opt/avtr1-code
  CUDA_VISIBLE_DEVICES=1 NOBILITY2_REFERENCE_CONFIG=/app/config/reference.json \
    /opt/venv-avtr1/bin/python /app/services/avtr1_service/server.py >> /var/log/avtr1.log 2>&1 &
}

echo "Starting main app on :7860..."
start_app

# --- 5. Watchdog: self-healing supervision -------------------------------
# Any service that dies gets restarted (with a log line), instead of the
# whole Space silently degrading until someone notices. This loop is the
# container's foreground process.
echo "All services up. Supervising."
while true; do
  sleep 5
  if ! pgrep -f "moshi.server" > /dev/null; then
    echo "$(date +%T) WATCHDOG: PersonaPlex died -- restarting (see /var/log/personaplex.log)"
    start_personaplex
  fi
  if ! pgrep -f "avtr1_service/server.py" > /dev/null; then
    echo "$(date +%T) WATCHDOG: AVTR-1 died -- restarting (see /var/log/avtr1.log)"
    start_avtr1
  fi
  if ! pgrep -f "uvicorn app:app" > /dev/null; then
    echo "$(date +%T) WATCHDOG: main app died -- restarting (see /var/log/app.log)"
    start_app
  fi
done
