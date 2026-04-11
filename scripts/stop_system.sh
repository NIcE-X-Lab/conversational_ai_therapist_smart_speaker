#!/bin/bash
echo "Stopping all CaiTI processes..."
fuser -k 8000/tcp || true
fuser -k 8001/tcp || true
pkill -f "python -u main.py" || true
pkill -f "python main.py" || true
echo "All processes stopped."
