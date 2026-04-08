#!/bin/bash
echo "Stopping all CaiTI processes..."
fuser -k 8000/tcp || true
fuser -k 8001/tcp || true
pkill -f "LLM_therapist_Application.py" || true
pkill -f "services/speech_service.py" || true
echo "All processes stopped."
