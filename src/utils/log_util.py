"""Utility helper providing standard standardized colored logger instances.

Console output modes
--------------------
CONSOLE_LOG_LEVEL env var sets the floor for console output across every
logger (default INFO). The file handler always stays at DEBUG.

CLINICIAN_LOG_MODE=1 applies a **tag-based** console filter so the
clinician only sees clinically-meaningful events. The file handler
remains unfiltered so every DEBUG line is retained for post-hoc review.

Rules (applied to the console handler only):
  1. WARNING and above always pass (even without a tag).
  2. At INFO, a message reaches the console only if it starts with one
     of the approved clinical-event tags in `_CLINICAL_TAGS`:
       [SESSION]     session lifecycle (start, end, subject identity)
       [PIPELINE]    explicit stage entry (Response Analyzer, RV, CBT, etc.)
       [RL]          Q-learning: state, action chosen, Q-update, reward
       [DLA]         Response Analyzer (Dim, Score) classifications
       [RV]          Reflection-Validation Reasoner / Guide / Validator
       [CBT]         CBT stage progression + decisions
       [QUESTIONER]  ask_question / evaluate_result milestones
       [INTERMISSION] screening/breathing/music activity transitions
       [USER]        user transcript (redacted if REDACT_PII=1)
       [AGENT]       agent utterance
       [SAFETY]      crisis override / safety-resources delivery
       [PHQ4]        PHQ-4 / GAD-2 screening answers + flags
       [SCORE]       per-dim score writes
  3. INFO messages without an approved tag stay in the file only.

Operators can extend the tag set per trial via CLINICIAN_LOG_TAGS
(comma-separated).
"""
import os
import re
import datetime
import coloredlogs
import logging
import sys
from src.utils.config_loader import LOG_DIR
# Ensure the log directory exists before logging
# Use umask to set directory permissions to 0777 for compatibility
if not os.path.exists(LOG_DIR):
    original_umask = os.umask(0)  # Set umask to 0 to allow full permissions
    os.makedirs(LOG_DIR, 0o777)  # Create the log directory with 0777 permissions
    os.umask(original_umask)  # Restore the original umask

_GLOBAL_LOG_FILE = os.environ.get("LOG_FILE")  # 可通过入口统一指定
_GLOBAL_FILE_HANDLER = None
_LOG_FORMAT = '%(asctime)s arth-desktop %(name)s[%(process)d] %(levelname)s %(message)s'

# ── Clinician console filter ─────────────────────────────────────────────
# Clinical-event tag registry. An INFO message reaches the clinician
# console only if its text begins with one of these bracketed tags.
# Pipeline-stage markers, RL traces, CBT/RV/DLA events, user/agent
# turns, and safety broadcasts all carry these tags; diagnostic chatter
# does not, so it stays in the file only.
_CLINICAL_TAGS: set[str] = {
    "[SESSION]",
    "[PIPELINE]",
    "[RL]",
    "[DLA]",
    "[RV]",
    "[CBT]",
    "[QUESTIONER]",
    "[INTERMISSION]",
    "[USER]",
    "[AGENT]",
    "[TTS]",       # every spoken utterance (bridge phrases, greetings, breathing, goodbye)
    "[SAFETY]",
    "[PHQ4]",
    "[SCORE]",
    "[GREETING]",
    "[CLOSING]",
}

# Cheap prefix check — any leading whitespace is tolerated. We also
# allow tags to appear immediately after a "<Label>: " prefix when the
# logger name has been quoted inline (e.g. "HandlerRL: [RL] ...").
_TAG_PATTERN = re.compile(
    r"^\s*(?:[A-Za-z0-9_\-\.]+:\s*)?(\[[A-Z0-9_ -]+\])"
)


# Section headers printed when the clinical tag changes. Maps the first
# tag in a consecutive run to a human-readable banner. Repeat tags
# within a section don't re-emit a header.
def _section_for(tag: str, message: str) -> str | None:
    """Return a banner to emit when a new clinical section begins."""
    m = message or ""
    if tag == "[SESSION]":
        lower = m.lower()
        if "wake phrase" in lower or "wake command" in lower:
            return "[SESSION START]"
        if "onboarded" in lower:
            return None  # already inside SESSION START
        if "clinical pipeline starting" in lower:
            return None  # same session, already banner'd
        if "clinical pipeline finished" in lower or "end signal" in lower or "ending" in lower:
            return "[SESSION END]"
        return None
    if tag == "[RL]":
        if "q-table seeded" in m.lower() or "no prior q-table" in m.lower() or "warm-start" in m.lower():
            return "[INITIALIZATION & RL SEEDING]"
        if "action chosen" in m.lower():
            return "[NEXT TURN SELECTION]"
        if "reward this turn" in m.lower():
            return None
        if "persisted longitudinal" in m.lower() or "updated q-table" in m.lower() or "created initial q-table" in m.lower():
            return "[POST-SESSION PERSISTENCE]"
        return None
    if tag == "[PIPELINE]":
        low = m.lower()
        if "screening loop starting" in low:
            return "[SCREENING LOOP]"
        if "screening loop complete" in low:
            return "[CBT PROTOCOL]"
        if "cbt protocol starting" in low:
            return "[CBT PROTOCOL]"
        if "response analyzer" in low:
            return "[RESPONSE ANALYSIS & SCORING]"
        if "reflection-validation" in low:
            return "[REFLECTION-VALIDATION]"
        if "report/notes csvs written" in low or "cbt protocol completed" in low:
            return "[POST-SESSION PERSISTENCE]"
        return None
    if tag == "[QUESTIONER]":
        if "starting turn" in m.lower():
            return None  # part of NEXT TURN SELECTION
        return None
    if tag == "[INTERMISSION]" or tag == "[PHQ4]":
        return "[INTERMISSION & PHQ-4 GATING]"
    if tag == "[TTS]":
        # Spoken utterances that aren't also [AGENT] (bridge phrases,
        # onboarding handshake, breathing scripts, goodbye). No banner —
        # they inherit whatever section is active when they land.
        return None
    if tag == "[CBT]":
        low = m.lower()
        if "stage 1" in low:
            return "[CBT STAGE 1: RECOGNIZE]"
        if "stage 2" in low:
            return "[CBT STAGE 2: CHALLENGE]"
        if "stage 3" in low:
            return "[CBT STAGE 3: REFRAME]"
        if "dimension selected" in low:
            return "[CBT DIMENSION SELECTION]"
        return None
    if tag == "[SAFETY]":
        return "[SAFETY OVERRIDE]"
    if tag == "[CLOSING]":
        return "[SESSION CLOSING]"
    return None


def _parse_env_tags(var_name: str, default: set[str]) -> set[str]:
    """Extend the tag set via a comma-separated env var (CLINICIAN_LOG_TAGS)."""
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return set(default)
    extra = {s.strip() for s in raw.split(",") if s.strip()}
    # Normalise: ensure each token is wrapped in brackets and uppercased.
    norm = set()
    for t in extra:
        t = t.upper()
        if not t.startswith("["):
            t = "[" + t
        if not t.endswith("]"):
            t = t + "]"
        norm.add(t)
    return set(default) | norm


_CLINICIAN_MODE = os.environ.get("CLINICIAN_LOG_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


class _ClinicianConsoleFilter(logging.Filter):
    """Console-only filter that passes only tagged clinical events at INFO.

    Applied as a handler-level filter on the StreamHandler created in
    `get_logger`. The file handler is NOT touched — it still receives
    the full DEBUG-level stream for post-session forensic review.

    Rules:
      1. Records at WARNING or above ALWAYS pass (resource warnings, DB
         errors, GPIO failures, etc. still reach the clinician even if
         the module doesn't use a tag).
      2. At INFO and below, the record's message must start with one of
         the approved clinical tags. Untagged INFO chatter (LLM heartbeat,
         VRAM handoff, audio hygiene, resource audits) stays in the file.
    """

    def __init__(self, tags: set[str]):
        super().__init__()
        self.tags = tags

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        m = _TAG_PATTERN.match(msg)
        if not m:
            return False
        return m.group(1) in self.tags


class _ClinicianConsoleFormatter(logging.Formatter):
    """Console formatter used only in CLINICIAN_LOG_MODE.

    Strips the normal `YYYY-MM-DD HH:MM:SS host name[pid] LEVEL ...`
    preamble — the clinician view is a clean clinical trace, not a
    system log.

    Prints a section banner (blank line + `[SECTION NAME]` + blank line)
    the first time a new clinical section begins, matching the
    hand-curated trace the clinician specified. Banners are global to
    the console handler so cross-logger transitions (HandlerRL -> CBT,
    CBT -> RL, etc.) still emit the right header.

    WARNING and above are prefixed with `!` so the clinician can spot
    a hardware/runtime issue in an otherwise clean trace.
    """

    def __init__(self):
        super().__init__()
        self._last_section: str | None = None

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        m = _TAG_PATTERN.match(msg)
        tag = m.group(1) if m else None

        prefix = ""
        if tag:
            section = _section_for(tag, msg)
            if section and section != self._last_section:
                # Emit a section banner and remember it.
                self._last_section = section
                prefix = f"\n{section}\n"

        # Strip any leading "LoggerName: " the caller may have prepended.
        if m and m.group(0) != msg[: len(m.group(0))]:
            pass  # pattern matched later in string — shouldn't happen
        if m:
            # Slice from the tag onward so "HandlerRL: [RL] ..." becomes "[RL] ..."
            msg = msg[m.start(1):]

        if record.levelno >= logging.WARNING:
            return f"{prefix}! {record.levelname}: {msg}"
        return f"{prefix}{msg}"


_CLINICIAN_FILTER = _ClinicianConsoleFilter(
    tags=_parse_env_tags("CLINICIAN_LOG_TAGS", _CLINICAL_TAGS),
)
_CLINICIAN_FORMATTER = _ClinicianConsoleFormatter()

def _ensure_global_file_handler():
    '''
    Ensure the global file handler is created.
    '''
    global _GLOBAL_FILE_HANDLER, _GLOBAL_LOG_FILE
    if _GLOBAL_FILE_HANDLER is None:
        if not _GLOBAL_LOG_FILE:
            _GLOBAL_LOG_FILE = os.path.join(
                LOG_DIR,
                datetime.datetime.now().strftime("output_%y%m%d_%H%M%S.log")
            )
        fh = logging.FileHandler(_GLOBAL_LOG_FILE)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        _GLOBAL_FILE_HANDLER = fh
    return _GLOBAL_FILE_HANDLER

def get_logger(name, file=None, file_handler=None):
    """
    Create and return a logger with both stream and file handlers.
    If file_handler is provided, it will be used for file logging.
    Otherwise, a new FileHandler will be created using the given file path or a timestamped default.
    Coloredlogs is used to enhance terminal output readability.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # file 仍写 DEBUG

    # 控制台日志级别来自环境变量，默认 DEBUG
    console_level_name = os.environ.get("CONSOLE_LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, console_level_name, logging.INFO)

    # 避免重复添加 handler
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        # Clinician-mode writes to STDOUT so native C++ libraries (LiteRT,
        # XNNPack, TensorFlow Lite) — which spew to STDERR — can be
        # redirected to the forensic file without polluting the clean
        # clinical trace. Non-clinician mode stays on stderr for legacy
        # behaviour.
        c_handler = logging.StreamHandler(sys.stdout if _CLINICIAN_MODE else sys.stderr)
        c_handler.setLevel(console_level)
        if _CLINICIAN_MODE:
            c_handler.addFilter(_CLINICIAN_FILTER)
            c_handler.setFormatter(_CLINICIAN_FORMATTER)
        logger.addHandler(c_handler)

    # 文件 handler 仍为 DEBUG 级别
    if file_handler is not None:
        f_handler = file_handler
    else:
        if file is not None:
            f_handler = logging.FileHandler(file)
            f_handler.setLevel(logging.DEBUG)
            f_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        else:
            f_handler = _ensure_global_file_handler()

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(f_handler)

    # coloredlogs installs its own StreamHandler+Formatter, which would
    # overwrite our clinician-clean format. Only install it when NOT in
    # clinician mode so the hand-curated clinical trace wins.
    if not _CLINICIAN_MODE:
        coloredlogs.install(level=console_level, logger=logger, fmt=_LOG_FORMAT)
    else:
        # Belt-and-suspenders: if any prior import already ran
        # coloredlogs.install on this logger, reapply our filter +
        # formatter to any new StreamHandler it created. File handler
        # stays on _LOG_FORMAT for full forensic detail.
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                if not any(isinstance(f, _ClinicianConsoleFilter) for f in h.filters):
                    h.addFilter(_CLINICIAN_FILTER)
                h.setFormatter(_CLINICIAN_FORMATTER)

    return logger


