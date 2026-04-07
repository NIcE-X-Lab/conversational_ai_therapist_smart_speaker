#!/bin/bash
set -e

# Configuration
LOCAL_DIR="$(pwd)"
PARENT_DIR="$(dirname "$LOCAL_DIR")"
ENV_FILE="$PARENT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Error: .env file not found in parent directory. Please ensure it exists with JETSON_HOST, JETSON_PASSWORD, OPENAI_MODEL."
    exit 1
fi

PROJECT_DIR="~/project"
MODEL_NAME=${OPENAI_MODEL:-"llama3.2-caiti"}

echo "🚀 Deploying Eric's Model to Jetson ($JETSON_HOST)..."

# 1. Sync the llm_model folder
echo "Syncing model files via rsync..."
rsync -avz --progress \
    $LOCAL_DIR/llm_model/ $JETSON_HOST:$PROJECT_DIR/eric_test/llm_model/

# 2. Remote execution: Build Ollama model and start the backend
echo "Running remote build and starting server..."
ssh $JETSON_HOST << EOF
    set -e
    echo "Connected to Jetson. Building Ollama model..."
    
    cd $PROJECT_DIR/eric_test/llm_model
    
    # Check if Modelfile exists and build
    if [ -f "Modelfile" ]; then
        echo "Found Modelfile. Running ollama create $MODEL_NAME..."
        # NOTE: Assumes ollama CLI is installed on Jetson
        ollama create $MODEL_NAME -f Modelfile
    else
        echo "No Modelfile found. Skipping custom build."
    fi
    
    # Navigate to project root and start the backend
    cd $PROJECT_DIR
    echo "Stopping existing services..."
    fuser -k 8000/tcp || true
    
    echo "Starting Dialogue Engine Backend..."
    source .venv/bin/activate
    export $(grep -v '^#' ".env" | xargs)
    nohup python -u LLM_therapist_Application.py > backend_session.log 2>&1 &
    BACKEND_PID=\$!
    echo "Backend Server running on PID: \$BACKEND_PID"
    
    echo "✅ Remote Backend Started."
EOF

echo ""
echo "🎉 Deployment and Backend Startup Successful!"
echo "The Jetson backend is now listening for remote connections."
echo ""
echo "Next Step: On this laptop, run the local speech client pointing to Jetson's IP."
echo "Command: python remote_speech_client.py"
