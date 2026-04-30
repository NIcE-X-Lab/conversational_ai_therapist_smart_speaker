"""Speech Emotion Recognition — future drop-in point.

Status: STUB.  The legacy `light_mfcc_rf` backend was removed during
research alignment (see `src/models/stt.py` header).  Pipeline plumbing
downstream of this class is intact and ready — `io_record.get_answer()`
and `get_resp_log()` both unpack `detected_emotion` from the STT
JSON envelope into `_LAST_USER_EMOTION`, which then rides the dossier
and the `ser_metrics["emotion_tag"]` channel to clinical analysis.

Drop-in contract
----------------
To land a real SER model in the future (e.g. a pruned fork of
`j-hartmann/emotion-english-distilroberta-base`, a wav2vec-2.0 IEMOCAP
head, or the `EmotionHeads` from xxue752-nz/emo_module):

  1. Implement `SERGenerator.detect(wav_path) -> str` returning a short
     tag (we recommend one of: "anxious", "sad", "angry", "happy",
     "calm", "neu").  Keep the vocabulary small and stable — downstream
     LLM prompts will treat this as a category, not free-form text.

  2. Implement `suspend()` / `resume()` to release and re-load the
     model weights on LLM-turn boundaries.  Follow the pattern in
     `STTGenerator.suspend/resume` so the STT-style VRAM handoff in
     `speech_service.py` keeps working.

  3. Flip `ser.ser_enabled: true` in config.yaml (or export
     `SER_ENABLED=true`).  `STTGenerator.transcribe()` will then call
     `detect()` and substitute the result into the JSON envelope.

  4. Validate under the 8 GB Jetson budget using the feasibility
     memory-check pattern from the `emo_module` audit (see chat log
     2026-04-30) — any SER that needs >400 MB RAM resident AT THE SAME
     TIME as Gemma will break the existing VRAM handoff contract.

Design note: SER is intentionally a separate class (not folded into
`STTGenerator`) so a new backend can hold its own weights, its own
suspend state, and its own subprocess / sidecar lifecycle without
leaking into the transcription path.
"""

from src.utils.config_loader import SER_ENABLED
from src.utils.log_util import get_logger

logger = get_logger("SERGenerator")

# The stub's "detection" output.  Downstream consumers treat "neu"
# (legacy three-letter tag) and "Neutral" (io_record default) as
# equivalent; we emit "neu" here so existing telemetry / dossier
# analysis paths continue to see exactly what they saw before the
# backend was removed.
_STUB_EMOTION_TAG = "neu"


class SERGenerator:
    """Plug-in seat for a future SER backend.

    When `SER_ENABLED=False` (default) this class is constructed but
    does no work — `is_ready` stays False and `detect()` returns the
    legacy stub tag so the downstream JSON envelope is unchanged.

    When SER is re-enabled in config, a real implementation must:
      - load a model in `__init__` (or lazy-load on first `detect()`),
      - return a short emotion tag from `detect(wav_path)`,
      - release weights on `suspend()` for VRAM headroom before LLM,
      - re-load on `resume()` before the next mic turn.
    """

    def __init__(self):
        self._enabled = bool(SER_ENABLED)
        self._model = None  # populated by a real backend
        if self._enabled:
            # A future backend lands here.  Until one does, log loudly
            # so an operator who flips the flag doesn't get silent
            # "neu" outputs and think the new SER is working.
            logger.warning(
                "[SER] ser.ser_enabled=true but no backend is installed. "
                "SERGenerator is a stub — every call returns 'neu'. "
                "Implement detect() in src/models/ser.py to land a real "
                "emotion detector, then validate against the Jetson "
                "memory budget before trusting downstream emotion tags."
            )
        else:
            logger.debug("[SER] Disabled (ser.ser_enabled=false) — stub active.")

    @property
    def is_ready(self) -> bool:
        """True when a real backend is loaded AND enabled."""
        return self._enabled and self._model is not None

    def detect(self, wav_path: str) -> str:
        """Return a short emotion tag for the given WAV file.

        Stub behaviour: always returns "neu".  Real backends MUST return
        a stable short tag (see class docstring for the recommended
        vocabulary).  Callers must tolerate "neu" as a no-signal
        fallback even once a real backend is in place (model failure,
        unreadable audio, etc. should not propagate as exceptions).
        """
        if not self._enabled or self._model is None:
            return _STUB_EMOTION_TAG
        # Real backend lands here — wrap in try/except so a decode
        # failure never breaks the transcription path.
        try:
            # return self._model.predict(wav_path)
            return _STUB_EMOTION_TAG
        except Exception as e:
            logger.warning(f"[SER] detect() failed on {wav_path}: {e}. Returning 'neu'.")
            return _STUB_EMOTION_TAG

    def suspend(self):
        """Release SER weights before an LLM turn (mirror STT pattern).

        Stub is a no-op.  A real backend must free GPU/CPU memory here
        so the VRAM handoff in `src/services/speech_service.py` keeps
        Gemma's inference budget intact.
        """
        if not self._enabled or self._model is None:
            return
        # Real backend: `del self._model; self._model = None; gc.collect()`

    def resume(self):
        """Re-load SER weights after an LLM turn returns.

        Stub is a no-op.  A real backend must re-initialise the model
        here so the next mic window can run emotion detection.
        """
        if not self._enabled:
            return
        # Real backend: `self._model = <constructor>()` (idempotent).
