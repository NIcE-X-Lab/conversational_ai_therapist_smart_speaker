#!/bin/bash
set -e

# Configuration
JETSON_USER="arth"
JETSON_IP="152.23.251.49"
JETSON_HOST="$JETSON_USER@$JETSON_IP"
PROJECT_DIR="~/project"
LOCAL_DIR="$(pwd)"

echo "🚀 Starting Deployment to Jetson Orin Nano ($JETSON_IP)..."

# 1. SSH Setup
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "Generating SSH key..."
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
fi

echo "Copying SSH ID to Jetson (You may be asked for password 'carolina2026')..."
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
    if ! dpkg -s portaudio19-dev >/dev/null 2>&1; then
        echo "Installing portaudio19-dev (sudo required)..."
        echo "carolina2026" | sudo -S apt-get install -y portaudio19-dev app
    fi
    
    # Python Venv
    if [ ! -d ".venv" ]; then
        echo "Creating Python venv..."
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
echo "4. Run Backend: python LLM_therapist_Application.py"
echo "5. (Optional) Run Frontend in another terminal: cd frontend && npm install && npm run dev -- --host"
