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

# Config-level default subject_id is a *boot-time fallback* only. The real
# subject identity is captured per session during onboarding ("Who am I
# speaking with today?"), then composed into "<name>_<YYYYMMDD_HHMMSS>"
# inside io_record.init_record() so every session — even two for the same
# person — lands in its own file entry. reset_session() installs that
# composed id into io_record.SUBJECT_ID.
SUBJECT_ID = str(APP["subject_id"])

# Raw, unexpanded templates — filled in by format_result_paths() at the
# moment a session starts so artefacts are session-scoped.
REPORT_FILE_TEMPLATE = PATHS["report_file"]
NOTES_FILE_TEMPLATE = PATHS["notes_file"]


def _expand(path: str, subject_id: str = SUBJECT_ID) -> str:
    """Fill ${subject_id} in a config path with the composed session id."""
    return path.replace("${subject_id}", str(subject_id or SUBJECT_ID))


def format_result_paths(subject_id: str) -> tuple[str, str]:
    """Return (report_path, notes_path) for a given composed subject id.

    The `subject_id` passed in is expected to already carry the session
    timestamp (e.g. "alice_20260425_190621") — io_record.init_record is
    responsible for that composition. That single transform is what
    makes each session a distinct file entry.
    """
    return (
        _expand(REPORT_FILE_TEMPLATE, subject_id),
        _expand(NOTES_FILE_TEMPLATE, subject_id),
    )


DATA_DIR = _expand(PATHS["data_dir"])
LOG_DIR = _expand(PATHS["logs_dir"])
RESULT_DIR = _expand(PATHS["result_dir"])
QUESTION_LIB_FILENAME = _expand(PATHS["question_lib_filename"])
# REPORT_FILE / NOTES_FILE retain a boot-time fallback expansion so any
# diagnostic caller that imports them gets a valid (pre-session) path.
# All production writers go through io_record.REPORT_FILE / NOTES_FILE,
# which are rewritten at init_record() per onboarded subject.
REPORT_FILE = _expand(REPORT_FILE_TEMPLATE, SUBJECT_ID)
NOTES_FILE = _expand(NOTES_FILE_TEMPLATE, SUBJECT_ID)
RECORD_CSV = _expand(PATHS["record_csv"])

ITEM_N_STATES = int(RL["item_n_states"])
EPSILON = float(RL["epsilon"])
ALPHA = float(RL["alpha"])
GAMMA = float(RL["gamma"])
ITEM_IMPORTANCE = RL["item_importance"]
NUMBER_QUESTIONS = RL["number_questions"]

# Paper §5.1 runtime Rephraser: structural rewrite of the picked question
# variant before it is spoken. Flag defaults preserve paper behaviour when
# the key is absent from config.yaml.
REPHRASE_AT_RUNTIME = bool(RL.get("rephrase_at_runtime", True))
REPHRASE_PROBABILITY = float(RL.get("rephrase_probability", 0.95))

# Legacy-parity gates. All default False so the runtime flow mirrors
# the demo video; flip via config.yaml to re-enable the paper/research
# extensions one at a time.  See config.yaml for detailed descriptions
# and re-enablement checklists.
#
# Pipeline-shape gates (G5/G8/G9):
#   REASK_DIMENSION_N         — paper §4.2 Dimension_N re-ask
#   MULTI_DIM_BACKFILL_ENABLED — paper §4.1 multi-dim back-fill (2 passes)
#   REFLECTIVE_SUMMARIZER_ENABLED — paper §5.2 MI reflective summarizer
#
# Session-lifecycle gates (G11-G15):
#   WARM_START_ENABLED        — returning-user Q-table nudge + recall greeting
#   SESSION_ANALYSIS_ENABLED  — post-session SUMMARY + preferences + safety LLM pass
#   SOAP_REPORT_ENABLED       — SOAP-format clinician note at session end
#   DIMENSION_OPTOUTS_ENABLED — per-user `disabled_dim:<label>` preference masking
#   SESSION_CAP_ENABLED       — 60-minute hard session-length cap
REASK_DIMENSION_N = bool(RL.get("reask_dimension_n", False))
MULTI_DIM_BACKFILL_ENABLED = bool(RL.get("multi_dim_backfill_enabled", False))
REFLECTIVE_SUMMARIZER_ENABLED = bool(RL.get("reflective_summarizer_enabled", False))
WARM_START_ENABLED = bool(RL.get("warm_start_enabled", False))
SESSION_ANALYSIS_ENABLED = bool(RL.get("session_analysis_enabled", False))
SOAP_REPORT_ENABLED = bool(RL.get("soap_report_enabled", False))
DIMENSION_OPTOUTS_ENABLED = bool(RL.get("dimension_optouts_enabled", False))
SESSION_CAP_ENABLED = bool(RL.get("session_cap_enabled", False))

# Per-dimension reward aggregation mode: "mean" = paper §5.1 / legacy
# prototype arithmetic mean; "hybrid" = (max+mean)/2, our default, keeps
# sensitivity to high-severity single segments. Any unknown value falls
# back to "hybrid".
_reward_mode_raw = str(RL.get("reward_mode", "hybrid")).strip().lower()
REWARD_MODE = _reward_mode_raw if _reward_mode_raw in ("mean", "hybrid") else "hybrid"


# LLM configuration — Gemma 4 E2B via LiteRT-LM (in-process inference)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-E2B-it")
LITERT_MODEL_PATH = os.environ.get(
    "LITERT_MODEL_PATH", "./models/litert/gemma-4-E2B-it.litertlm"
)
LITERT_BACKEND = os.environ.get("LITERT_BACKEND", "cpu").strip().lower()
# Engine-level capacity. litert_lm.Engine takes `max_num_tokens` at
# construction (applies to every call until teardown); there is no
# per-call override on send_message in this version of the library.
# We set it generously (4096) so paragraph-length Validator / CBT-Guide
# outputs — the ones that mirror the demo's "It makes sense that you
# always forget... A few steps that can help... Ask your prescriber..."
# style — are never clipped mid-sentence.  Smaller roles (Reasoner,
# Analyzer) self-limit via their prompts ("DECISION: 0/1 only"), so the
# generous ceiling does not cost latency there.
LITERT_CONTEXT_LENGTH = int(os.environ.get("LITERT_CONTEXT_LENGTH", "4096"))
LITERT_MAX_TOKENS = int(os.environ.get("LITERT_MAX_TOKENS", "4096"))

# Informational per-role soft budgets. Today Gemma-4-E2B is the sole
# backend and cannot take per-call max-token overrides, so these values
# are not wired through to the engine — they document intended output
# length per role for future multi-engine deployments where the map in
# `src/models/llm_client.py::ROLE_MODEL_MAP` is split across models.
#
# Budgets are chosen to match the demo video's observed output length:
#   RV_VALIDATOR    : 3-5 sentence MI reflection + concrete strategies
#   RV_GUIDE        : long enumerations ("you think X; you fear Y; ...")
#   CBT_GUIDE       : long enumerations for Stage-1 unhelpful thoughts
#   CBT_REASONER    : single "DECISION: 0/1" line
#   RV_REASONER     : single "DECISION: 0/1" line
#   ANALYZER        : terse (dim, score) pair
#   REPHRASER       : 1-2 sentence structural rewrite
#   REFLECTIVE_SUMM : 1st->3rd person restatement, 1 sentence
#   GENERAL         : greeting / closing / session-analysis summary
ROLE_MAX_TOKENS: dict[str, int] = {
    "rv_validator": 512,
    "rv_guide": 512,
    "cbt_guide": 512,
    "cbt_reasoner": 96,
    "rv_reasoner": 96,
    "analyzer": 96,
    "rephraser": 160,
    "reflective_summarizer": 160,
    "general": 400,
}

OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "90"))

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

# TTS
TTS = _CFG.get("tts", {})
TTS_MODEL_PATH = os.environ.get("TTS_MODEL_PATH", TTS.get("model_path", "./models/piper/en_US-amy-medium.onnx"))
TTS_EXECUTABLE = TTS.get("executable_path", "piper")
TTS_LENGTH_SCALE = float(os.environ.get("TTS_LENGTH_SCALE", "0.8"))
TTS_SENTENCE_SILENCE = float(os.environ.get("TTS_SENTENCE_SILENCE", "1.5"))
# Second voice for intermission-only TTS (lead-in, breathing, music,
# paired screening).  Blank string disables the second voice so every
# utterance uses TTS_MODEL_PATH.  Piper is subprocess-invoked per
# utterance, so there is no resident-memory cost for keeping two
# voices on disk — only disk space (~63 MB for Alan medium).
TTS_INTERMISSION_MODEL_PATH = os.environ.get(
    "TTS_INTERMISSION_MODEL_PATH", TTS.get("intermission_model_path", "")
).strip()

# Database
DATABASE = _CFG.get("database", {})
DB_PATH = DATABASE.get("db_path", "data/therapist.db")

# Speech Emotion Recognition (SER)
# Disabled by default — see config.yaml `ser:` block for the drop-in
# contract.  STTGenerator checks SER_ENABLED at init time and only
# constructs the SERGenerator when the flag is true, so the "off"
# state truly loads no additional model.
SER_CFG = _CFG.get("ser", {})
SER_ENABLED = os.environ.get(
    "SER_ENABLED", str(SER_CFG.get("ser_enabled", False))
).strip().lower() in {"1", "true", "yes", "on"}

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


