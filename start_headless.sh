#!/bin/bash
set -e

# Kill existing
echo "Stopping existing services..."
fuser -k 8000/tcp || true
fuser -k 8001/tcp || true
pkill -f "services/speech_service.py" || true

# Start Ollama LLM Service (if not already running)
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama service..."
    nohup ollama serve > ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "Ollama PID: $OLLAMA_PID"
    # Wait up to 15s for Ollama to become ready
    for i in $(seq 1 15); do
        if curl -s --max-time 1 http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama is ready."
            
            # Ensure fine-tuned model exists
            if ! ollama list | grep -q "llama3.2-caiti"; then
                echo "Creating llama3.2-caiti model from Modelfile..."
                ollama create llama3.2-caiti -f Modelfile
            else
                echo "✅ llama3.2-caiti model found."
            fi
            break
        fi
        echo "Waiting for Ollama... ($i/15)"
        sleep 1
    done
else
    echo "✅ Ollama already running."
fi

# Start Backend
echo "Starting Dialogue Engine (Server)..."
source .venv/bin/activate
if [ -f ".env" ]; then
    echo "Sourcing .env variables..."
    export $(grep -v '^#' ".env" | xargs)
fi
export DISABLE_INTERNAL_SPEECH=1
export CONSOLE_LOG_LEVEL=DEBUG
nohup python -u LLM_therapist_Application.py > backend_session.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for Backend to be ready
echo "Waiting for Backend to initialize..."
sleep 5

# Start Speech Service
echo "Starting Speech Service (Client)..."
export PYTHONPATH=.
nohup python -u services/speech_service.py > speech_service.log 2>&1 &
SPEECH_PID=$!
echo "Speech Service PID: $SPEECH_PID"

echo "Headless System Started!"
echo "Logs: tail -f backend_session.log speech_service.log"
