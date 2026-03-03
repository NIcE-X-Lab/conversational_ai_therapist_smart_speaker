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

PROJECT_DIR="~/project"

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
    $LOCAL_DIR/ $JETSON_HOST:$PROJECT_DIR/

# 3. Remote Setup Script
echo "Running remote setup..."
ssh $JETSON_HOST << EOF
    set -e
    echo "Connected to Jetson. Setting up environment..."
    
    # Create project dir if needed (rsync handles typically but good to ensure)
    mkdir -p $PROJECT_DIR
    cd $PROJECT_DIR
    
    # Install System Deps
    echo "Installing dependencies (sudo required)..."
    echo "$JETSON_PASSWORD" | sudo -S apt-get install -y portaudio19-dev python3-venv

    
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
    pip install faster-whisper webrtcvad pyaudio soundfile uvicorn fastapi python-multipart
    
    # Sanity Check
    echo "Running Sanity Check..."
    python sanity_check.py
    
    echo "✅ Remote Setup Complete."
EOF

echo "🎉 Deployment Successful!"
echo "To run the system:"
echo "1. SSH into Jetson: ssh $JETSON_HOST"
echo "2. Go to project: cd $PROJECT_DIR"
echo "3. Activate venv: source .venv/bin/activate"
echo "4. Run Headless System: ./start_headless.sh"
