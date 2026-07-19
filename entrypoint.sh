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
  /opt/venv-avtr1/bin/python scripts/build_engines.py
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
echo "Starting PersonaPlex service..."
cd /data/personaplex
/opt/venv-personaplex/bin/python -m moshi.server \
  --host 0.0.0.0 --port 8998 \
  --moshi-weight /data/personaplex/model.safetensors \
  --mimi-weight /data/personaplex/tokenizer-e351c8d8-checkpoint125.safetensors \
  --tokenizer /data/personaplex/tokenizer_spm_32k_3.model \
  --voice-prompt-dir /data/personaplex/voices/voices \
  --device cuda > /var/log/personaplex.log 2>&1 &

echo "Starting AVTR-1 rendering service..."
cd /opt/avtr1-code
NOBILITY2_REFERENCE_CONFIG=/app/config/reference.json \
  /opt/venv-avtr1/bin/python /app/services/avtr1_service/server.py > /var/log/avtr1.log 2>&1 &

echo "Waiting for model services to be ready..."
until curl -sf http://localhost:8998/ > /dev/null 2>&1; do sleep 2; done
echo "PersonaPlex ready."

echo "Starting main app on :7860..."
cd /app
exec /opt/venv-app/bin/python -m uvicorn app:app --host 0.0.0.0 --port 7860
