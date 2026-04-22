import os
from typing import Any, Dict

import yaml

_ROOT_DIR = os.path.abspath(".")
_CONFIG_PATH = os.path.join(_ROOT_DIR, "config.yaml")

def _load_yaml_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("config.yaml must contain a top-level mapping")
        return data

_CFG = _load_yaml_config()

APP = _CFG["app"]
PATHS = _CFG["paths"]
RL = _CFG["rl"]
OPENAI = _CFG["openai"]

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

OPENAI_BASE_URL = OPENAI.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1"))
OPENAI_MODEL = OPENAI["model"]
OPENAI_TEMPERATURE = float(OPENAI["temperature"])
OPENAI_MAX_TOKENS = int(OPENAI["max_tokens"])


