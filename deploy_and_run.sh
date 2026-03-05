#!/bin/bash
set -e

LOCAL_DIR="$(pwd)"

# Ensure deploy_to_jetson.sh exists
if [ ! -f "$LOCAL_DIR/deploy_to_jetson.sh" ]; then
    echo "Error: deploy_to_jetson.sh not found in the current directory."
    exit 1
fi

# Run the deployment script
echo "Step 1: Running deployment script..."
bash "$LOCAL_DIR/deploy_to_jetson.sh"

# Configuration from .env
if [ -f "$LOCAL_DIR/.env" ]; then
    export $(grep -v '^#' "$LOCAL_DIR/.env" | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

PROJECT_DIR="~/project"

# SSH and run the headless system
echo "Step 2: Starting the system on Jetson ($JETSON_HOST)..."
ssh -t -t -o StrictHostKeyChecking=no "$JETSON_HOST" "cd $PROJECT_DIR && echo 'Executing start_headless.sh...' && bash ./start_headless.sh && echo '' && echo '===================================================' && echo 'System started in background. Tailing live logs...' && echo 'Press [Ctrl+C] to stop tailing and return to local.' && echo '===================================================' && tail -f backend_session.log speech_service.log"

echo "Done! The system is deployed and running on the Jetson."
