# Eric's Local Test Environment

This folder is built to let you test your own **custom LLMs** on the Jetson while using your **laptop's microphone and speakers** for interaction. 

It splits the architecture in half: The heavy AI (Llama Backend) runs headless on the Jetson, and the Speech Service (VAD/STT/TTS) runs completely locally on your machine.

## Folder Structure
- `llm_model/`: Put your `Modelfile` or your `.gguf` weights in here. The deployment script will sync them to the Jetson.
- `deploy_and_run_server.sh`: Pushes your custom model to the Jetson, executes `ollama create` there, and then starts the `LLM_therapist_Application.py` server headless in the background.
- `remote_speech_client.py`: A local python script you run on your own laptop to talk to the Jetson backend.

---

## Workflow Instructions

### 1. Build Your Model
Drop your `Modelfile` into the `llm_model` directory. 
Make sure you update the `OPENAI_MODEL` name in your main `.env` file at the root of the project to match the model you are deploying.

### 2. Deploy and Start Jetson Backend
From inside this `eric_test` directory, run the deployment script:
```bash
./deploy_and_run_server.sh
```
*This will sync your model files over SSH, build the model on the Jetson using Ollama, and start the python FastAPI backend.*

### 3. Connect Locally
Make sure you have activated your local python workspace:
```bash
cd ..
source .venv/bin/activate
cd eric_test
```

Then run the speech client. When prompted, input the IP address of the Jetson.
```bash
python remote_speech_client.py
```
*Your laptop is now streaming STT inputs directly over WiFi to your custom model running on the Jetson, and playing the TTS responses back out through your laptop speakers!*
