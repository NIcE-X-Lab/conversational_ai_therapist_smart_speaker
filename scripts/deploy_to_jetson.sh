#!/bin/bash
set -e

LOCAL_DIR="$(pwd)"

# Configuration from .env
if [ -f "$LOCAL_DIR/.env" ]; then
    export $(grep -v '^#' "$LOCAL_DIR/.env" | xargs)
else
    echo "Error: .env file not found. Please create one with JETSON_USER, JETSON_IP, JETSON_HOST, JETSON_PASSWORD."
    exit 1
fi

PROJECT_DIR="project" # Relative to home on remote

echo "🚀 Starting Deployment to Jetson Orin Nano ($JETSON_IP)..."

# 1. SSH Setup
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "Generating SSH key..."
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
fi

echo "Copying SSH ID to Jetson (You may be asked for password if not already set up)..."
ssh-copy-id -o StrictHostKeyChecking=no $JETSON_HOST || echo "SSH key copy failed or key exists."

# 2. Sync Codebase
echo "Syncing codebase via rsync..."
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude 'node_modules' \
    --exclude '.git' \
    --exclude 'data/therapist.db' \
    $LOCAL_DIR/ $JETSON_HOST:~/$PROJECT_DIR/

# 3. Remote Setup Script
echo "Running remote setup..."
ssh $JETSON_HOST << EOF
    set -e
    REMOTE_PROJECT_DIR="\$HOME/$PROJECT_DIR"
    echo "Connected to Jetson. Setting up environment in \$REMOTE_PROJECT_DIR..."
    
    mkdir -p "\$REMOTE_PROJECT_DIR"
    cd "\$REMOTE_PROJECT_DIR"
    
    # Install System Deps
    echo "Installing dependencies (sudo required)..."
    echo "$JETSON_PASSWORD" | sudo -S apt-get install -y portaudio19-dev python3-venv python3-jetson-gpio

    # Python Venv
    if [ ! -f ".venv/bin/activate" ]; then
        echo "Creating Python venv..."
        rm -rf .venv
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    
    # Install Python Requirements
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Download Piper en_US-amy-medium voice model if not present
    echo "Checking for Piper voice model (en_US-amy-medium)..."
    PIPER_MODEL_DIR="\$REMOTE_PROJECT_DIR/models/piper"
    mkdir -p "\$PIPER_MODEL_DIR"
    if [ ! -f "\$PIPER_MODEL_DIR/en_US-amy-medium.onnx" ]; then
        echo "Downloading en_US-amy-medium Piper voice model..."
        BASE_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2"
        wget -q -O "\$PIPER_MODEL_DIR/en_US-amy-medium.onnx" \
            "\$BASE_URL/voice-en_US-amy-medium.onnx" || echo "⚠️  Model download failed — check network."
        wget -q -O "\$PIPER_MODEL_DIR/en_US-amy-medium.onnx.json" \
            "\$BASE_URL/voice-en_US-amy-medium.onnx.json" || true
        echo "✅ Piper voice model downloaded."
    else
        echo "✅ Piper voice model already present."
    fi

    # Backward-compatibility path expected by older env files
    mkdir -p "\$REMOTE_PROJECT_DIR/models"
    ln -sf "\$PIPER_MODEL_DIR/en_US-amy-medium.onnx" "\$REMOTE_PROJECT_DIR/models/en_US-amy-medium.onnx"
    ln -sf "\$PIPER_MODEL_DIR/en_US-amy-medium.onnx.json" "\$REMOTE_PROJECT_DIR/models/en_US-amy-medium.onnx.json" || true

    echo "✅ Verifying Python syntax..."
    python3 -m py_compile main.py && echo "✅ main.py syntax OK."
    
    echo "✅ Remote Setup Complete."
EOF

echo "🎉 Deployment Successful!"
echo "To run the system:"
echo "1. SSH into Jetson: ssh $JETSON_HOST"
echo "2. Go to project: cd ~/$PROJECT_DIR"
echo "3. Activate venv: source .venv/bin/activate"
echo "4. Run Headless System: bash ./scripts/start_headless.sh"
