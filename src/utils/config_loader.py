"""Utility helper for fetching and defaulting central .env configurations."""
import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

_ROOT_DIR = os.path.abspath(".")
_CONFIG_PATH = os.path.join(_ROOT_DIR, "config.yaml")
_ENV_PATH = os.path.join(_ROOT_DIR, ".env")

load_dotenv(_ENV_PATH)

def _load_yaml_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("config.yaml must contain a top-level mapping")
        return data

_CFG = _load_yaml_config()

_REQUIRED_SECTIONS = ("app", "paths", "rl")
for _sect in _REQUIRED_SECTIONS:
    if _sect not in _CFG:
        raise KeyError(f"config.yaml missing required top-level key: '{_sect}'")

APP = _CFG["app"]
PATHS = _CFG["paths"]
RL = _CFG["rl"]

SUBJECT_ID = str(APP["subject_id"])

def _expand(path: str) -> str:
    return path.replace("${subject_id}", SUBJECT_ID)

DATA_DIR = _expand(PATHS["data_dir"])
LOG_DIR = _expand(PATHS["logs_dir"])
RESULT_DIR = _expand(PATHS["result_dir"])
QUESTION_LIB_FILENAME = _expand(PATHS["question_lib_filename"])
REPORT_FILE = _expand(PATHS["report_file"])
NOTES_FILE = _expand(PATHS["notes_file"])
RECORD_CSV = _expand(PATHS["record_csv"])

ITEM_N_STATES = int(RL["item_n_states"])
EPSILON = float(RL["epsilon"])
ALPHA = float(RL["alpha"])
GAMMA = float(RL["gamma"])
ITEM_IMPORTANCE = RL["item_importance"]
NUMBER_QUESTIONS = RL["number_questions"]

_ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Ensure the openai-compat path ends with /v1
OPENAI_BASE_URL = _ollama_base.rstrip("/") + "/v1"
# LLM configuration (replaces OPENAI_MODEL)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma:2b")
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",")
    if m.strip()
]

# If using openai package to query ollama directly, we can define tokens here:
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "400"))
LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "90"))

# Ollama per-request safety controls to reduce Jetson memory pressure.
# num_ctx=256: Aggressive cap to fit the KV cache handshake into fragmented
# VRAM.  The 3B model's cudaMalloc was requesting ~1918 MiB at num_ctx=512,
# exceeding available contiguous memory.  256 halves the KV cache further
# and should bring the request down to ~1200 MiB.  Override with env var
# once memory headroom is confirmed.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "256"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", str(OPENAI_MAX_TOKENS)))
# Default to full GPU offload (num_gpu=99 = all layers).  CPU-only (num_gpu=0)
# causes 7-minute inference on Jetson and should only be used as a deliberate
# override for devices with no GPU.
OLLAMA_NUM_GPU = int(os.environ.get("OLLAMA_NUM_GPU", "99"))

OLLAMA_KEEP_ALIVE = int(os.environ.get("OLLAMA_KEEP_ALIVE", "0"))

# Zero-History diagnostic toggle: when enabled, conversation context and user
# history are stripped from LLM prompts.  If the crash still occurs on turn 1,
# the issue is *Initialization Peak* (Whisper + KV cache).  If it only occurs
# on turn 5+, the issue is *Context Bloat* (unbounded history injection).
DISABLE_CONTEXT_HISTORY = os.environ.get(
    "DISABLE_CONTEXT_HISTORY", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Audio
AUDIO = _CFG.get("audio", {})
AUDIO_SAMPLE_RATE = int(AUDIO.get("sample_rate", 16000))
AUDIO_CHANNELS = int(AUDIO.get("channels", 1))
AUDIO_CHUNK_SIZE = int(AUDIO.get("chunk_size", 1024))
AUDIO_VAD_AGGRESSIVENESS = int(AUDIO.get("vad_aggressiveness", 3))

# STT
STT = _CFG.get("stt", {})
STT_MODEL_PATH = os.environ.get("STT_MODEL", STT.get("model_path", "base.en"))
STT_DEVICE = STT.get("device", "cpu")
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", STT.get("compute_type", "int8"))
STT_BEAM_SIZE = int(os.environ.get("STT_BEAM_SIZE", str(STT.get("beam_size", 2))))
STT_BEST_OF = int(os.environ.get("STT_BEST_OF", str(STT.get("best_of", 1))))
STT_WITHOUT_TIMESTAMPS = os.environ.get(
    "STT_WITHOUT_TIMESTAMPS", str(STT.get("without_timestamps", True))
).strip().lower() in {"1", "true", "yes", "on"}

# Emotion Recognition (SER)
SER_BACKEND = os.environ.get("SER_BACKEND", "light_mfcc_rf").strip().lower()
SER_DEVICE = os.environ.get("SER_DEVICE", "cpu")

# TTS
TTS = _CFG.get("tts", {})
TTS_MODEL_PATH = os.environ.get("TTS_MODEL_PATH", TTS.get("model_path", "./models/piper/en_US-amy-medium.onnx"))
TTS_EXECUTABLE = TTS.get("executable_path", "piper")
TTS_LENGTH_SCALE = float(os.environ.get("TTS_LENGTH_SCALE", "0.8"))
TTS_SENTENCE_SILENCE = float(os.environ.get("TTS_SENTENCE_SILENCE", "1.5"))

# Database
DATABASE = _CFG.get("database", {})
DB_PATH = DATABASE.get("db_path", "data/therapist.db")

# Hardware Pins
PIN_LISTENING_LED = int(os.environ.get("PIN_LISTENING_LED", "18"))
PIN_BTN_START = int(os.environ.get("PIN_BTN_START", "11"))
PIN_BTN_END = int(os.environ.get("PIN_BTN_END", "13"))
PIN_BTN_OPT_OUT = int(os.environ.get("PIN_BTN_OPT_OUT", "15"))
PIN_BTN_4 = int(os.environ.get("PIN_BTN_4", "16"))
PIN_LISTENING_LED_ACTIVE_LOW = os.environ.get("PIN_LISTENING_LED_ACTIVE_LOW", "0").strip().lower() in {"1", "true", "yes", "on"}
PIN_BUTTONS_ACTIVE_LOW = os.environ.get("PIN_BUTTONS_ACTIVE_LOW", "1").strip().lower() in {"1", "true", "yes", "on"}
PIN_BTN_START_ACTIVE_LOW = os.environ.get("PIN_BTN_START_ACTIVE_LOW", "0").strip().lower() in {"1", "true", "yes", "on"}
PIN_BTN_END_ACTIVE_LOW = os.environ.get("PIN_BTN_END_ACTIVE_LOW", "0").strip().lower() in {"1", "true", "yes", "on"}
PIN_BTN_OPT_OUT_ACTIVE_LOW = os.environ.get("PIN_BTN_OPT_OUT_ACTIVE_LOW", "1").strip().lower() in {"1", "true", "yes", "on"}
PIN_BTN_4_ACTIVE_LOW = os.environ.get("PIN_BTN_4_ACTIVE_LOW", "1").strip().lower() in {"1", "true", "yes", "on"}


