"""Lightweight MFCC-based emotion classifier for memory-constrained devices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import soundfile as sf

from src.utils.log_util import get_logger

logger = get_logger("LightSER")


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (freq_hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.shape[0] / float(src_sr)
    dst_size = max(1, int(round(duration * dst_sr)))
    src_idx = np.arange(audio.shape[0], dtype=np.float32)
    dst_idx = np.linspace(0, audio.shape[0] - 1, num=dst_size, dtype=np.float32)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)


def _build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    n_bins = (n_fft // 2) + 1
    mel_points = np.linspace(
        _hz_to_mel(np.array([f_min], dtype=np.float32))[0],
        _hz_to_mel(np.array([f_max], dtype=np.float32))[0],
        n_mels + 2,
        dtype=np.float32,
    )
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left = int(bin_points[i - 1])
        center = int(bin_points[i])
        right = int(bin_points[i + 1])

        if center <= left:
            center = min(left + 1, n_bins - 1)
        if right <= center:
            right = min(center + 1, n_bins - 1)

        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / float(right - center)

    return fb


def _build_dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float32)
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    dct = np.sqrt(2.0 / n_mels) * np.cos((np.pi / n_mels) * (n + 0.5) * k)
    dct[0, :] *= np.sqrt(0.5)
    return dct.astype(np.float32)


@dataclass(frozen=True)
class _TreeRule:
    feature_idx: int
    threshold: float
    left_label: str
    right_label: str


class LightweightRandomForestSER:
    """
    Tiny random-forest-style classifier over MFCC and prosody features.
    Designed for low-RAM embedded deployments.
    """

    LABELS = ("neu", "hap", "sad", "ang")

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        n_fft: int = 512,
        n_mels: int = 26,
        n_mfcc: int = 13,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_size = max(64, int(sample_rate * frame_ms / 1000.0))
        self.hop_size = max(32, int(sample_rate * hop_ms / 1000.0))
        self.n_fft = int(n_fft)
        self.n_mels = int(n_mels)
        self.n_mfcc = int(n_mfcc)

        self._mel_fb = _build_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=20.0,
            f_max=self.sample_rate / 2.0,
        )
        self._dct = _build_dct_matrix(self.n_mfcc, self.n_mels)
        self._rules = self._build_rules()
        logger.info("Lightweight MFCC-RF emotion model initialized.")

    def _build_rules(self) -> List[_TreeRule]:
        # Feature layout: [mfcc_mean(13), mfcc_std(13), rms_mean, rms_std, zcr_mean, zcr_std]
        #
        # Anger bias fix: MFCC std thresholds (features 13-15) were too low,
        # causing normal conversational speech to trigger 3+ anger votes.
        # Raised thresholds so only genuinely high-variance (loud/sharp) speech
        # triggers anger.  ZCR anger thresholds also raised.
        return [
            _TreeRule(26, 0.030, "sad", "neu"),     # RMS mean: low energy → sad
            _TreeRule(26, 0.055, "neu", "hap"),     # RMS mean: high energy → hap
            _TreeRule(27, 0.020, "sad", "neu"),     # RMS std: low variance → sad
            _TreeRule(28, 0.120, "neu", "ang"),     # ZCR mean: raised from 0.090
            _TreeRule(29, 0.050, "neu", "ang"),     # ZCR std: raised from 0.030
            _TreeRule(0, -32.0, "sad", "neu"),      # MFCC0 mean: very low → sad
            _TreeRule(1, -2.0, "sad", "neu"),       # MFCC1 mean: low → sad
            _TreeRule(2, 1.5, "ang", "neu"),        # MFCC2 mean: low → ang
            _TreeRule(13, 22.0, "ang", "neu"),      # MFCC0 std: raised from 14.0
            _TreeRule(14, 16.0, "ang", "neu"),      # MFCC1 std: raised from 9.0
            _TreeRule(15, 14.0, "ang", "neu"),      # MFCC2 std: raised from 8.0
            _TreeRule(4, 2.5, "hap", "neu"),        # MFCC4 mean: positive → hap
            _TreeRule(5, 1.8, "hap", "neu"),        # MFCC5 mean: positive → hap
            _TreeRule(6, -0.5, "sad", "hap"),       # MFCC6 mean: low → sad
            _TreeRule(28, 0.060, "sad", "hap"),     # ZCR mean: low → sad
        ]

    def _frame_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] < self.frame_size:
            pad = np.zeros(self.frame_size - audio.shape[0], dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=0)

        remainder = (audio.shape[0] - self.frame_size) % self.hop_size
        if remainder != 0:
            pad = np.zeros(self.hop_size - remainder, dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=0)

        frames = []
        for start in range(0, audio.shape[0] - self.frame_size + 1, self.hop_size):
            frames.append(audio[start : start + self.frame_size])
        return np.stack(frames, axis=0).astype(np.float32)

    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32, copy=False)

        if audio.size == 0:
            return np.zeros((self.n_mfcc * 2) + 4, dtype=np.float32)

        audio = _resample_linear(audio, sample_rate, self.sample_rate)

        # ── Auto-gain normalization (RMS-based) ───────────────────────────
        # Peak normalization alone is insufficient on Jetson where mic gain
        # varies.  RMS normalization ensures consistent feature magnitudes
        # regardless of input gain, preventing the "always angry" bias.
        rms_level = np.sqrt(np.mean(audio ** 2) + 1e-9)
        target_rms = 0.1  # target RMS for normalized speech
        if rms_level > 1e-6:
            audio = audio * (target_rms / rms_level)
        # Clip to [-1, 1] after RMS normalization
        audio = np.clip(audio, -1.0, 1.0)

        # Simple pre-emphasis to retain high-frequency cues.
        if audio.shape[0] > 1:
            audio = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]], axis=0)

        frames = self._frame_audio(audio)
        windowed = frames * np.hamming(self.frame_size).astype(np.float32)

        spectrum = np.fft.rfft(windowed, n=self.n_fft, axis=1)
        power = (np.abs(spectrum) ** 2).astype(np.float32) / float(self.n_fft)

        mel_energy = np.dot(power, self._mel_fb.T)
        log_mel = np.log(mel_energy + 1e-6)
        mfcc = np.dot(log_mel, self._dct.T)

        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_std = np.std(mfcc, axis=0)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)
        zcr = np.mean((frames[:, :-1] * frames[:, 1:]) < 0, axis=1).astype(np.float32)

        features = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                np.array([
                    float(np.mean(rms)),
                    float(np.std(rms)),
                    float(np.mean(zcr)),
                    float(np.std(zcr)),
                ], dtype=np.float32),
            ],
            axis=0,
        )
        return features.astype(np.float32)

    # Minimum RMS threshold below which audio is considered silence/noise
    # and SER is skipped entirely (returns "neu").
    _SILENCE_RMS_THRESHOLD = 0.005

    # ── Adaptive anger bias detection ──────────────────────────────────────
    # If anger is classified > _ANGER_BIAS_WINDOW times out of the last N
    # calls, dynamically raise MFCC std thresholds by 10% to suppress the
    # bias.  This self-corrects on devices where mic gain or ambient noise
    # consistently inflates variance features.
    _ANGER_BIAS_WINDOW = 10
    _ANGER_BIAS_TRIGGER = 7  # 7/10 = 70% anger → bias detected

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _init_bias_tracker(self):
        """Lazy-init for the rolling anger-bias tracker."""
        if not hasattr(self, "_recent_labels"):
            self._recent_labels: List[str] = []
            self._anger_calibrations = 0

    def _check_anger_bias(self):
        """If anger dominates recent classifications, raise MFCC std thresholds 10%."""
        self._init_bias_tracker()
        if len(self._recent_labels) < self._ANGER_BIAS_WINDOW:
            return

        recent = self._recent_labels[-self._ANGER_BIAS_WINDOW:]
        anger_count = sum(1 for l in recent if l == "ang")

        if anger_count >= self._ANGER_BIAS_TRIGGER:
            self._anger_calibrations += 1
            # Raise MFCC std thresholds (features 13, 14, 15) by 10%
            for idx, rule in enumerate(list(self._rules)):
                if rule.feature_idx in (13, 14, 15) and rule.left_label == "ang":
                    new_thresh = round(rule.threshold * 1.10, 1)
                    old_thresh = rule.threshold
                    # Replace with adjusted rule (frozen dataclass, must recreate)
                    self._rules[idx] = _TreeRule(
                        feature_idx=rule.feature_idx,
                        threshold=new_thresh,
                        left_label=rule.left_label,
                        right_label=rule.right_label,
                    )
                    logger.warning(
                        f"[SER CALIBRATE #{self._anger_calibrations}] "
                        f"Anger bias detected ({anger_count}/{self._ANGER_BIAS_WINDOW}). "
                        f"Feature {rule.feature_idx} threshold: {old_thresh} -> {new_thresh}"
                    )
            # Reset window after calibration
            self._recent_labels.clear()

    def classify_file(self, audio_path: str) -> str:
        try:
            audio, sample_rate = sf.read(audio_path, always_2d=False)

            # ── Silence gate: skip SER on near-silent / noise-only buffers ──
            if audio.ndim > 1:
                mono = np.mean(audio, axis=1)
            else:
                mono = audio
            raw_rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2) + 1e-9))
            if raw_rms < self._SILENCE_RMS_THRESHOLD:
                logger.info(
                    f"SER silence gate: RMS={raw_rms:.5f} < "
                    f"{self._SILENCE_RMS_THRESHOLD}. Tagging as 'neu'."
                )
                return "neu"

            features = self._extract_features(audio, sample_rate)

            votes: Dict[str, int] = {label: 0 for label in self.LABELS}
            for rule in self._rules:
                label = rule.left_label if features[rule.feature_idx] <= rule.threshold else rule.right_label
                votes[label] += 1

            winner = max(votes, key=lambda k: (votes[k], k == "neu"))

            # Diagnostic: log feature values and votes for debugging anger bias
            logger.debug(
                f"SER features: rms_mean={features[26]:.4f} rms_std={features[27]:.4f} "
                f"zcr_mean={features[28]:.4f} zcr_std={features[29]:.4f} | "
                f"Votes: {votes} -> {winner}"
            )

            # ── Track result for adaptive anger bias detection ──
            self._init_bias_tracker()
            self._recent_labels.append(winner)
            self._check_anger_bias()

            return winner
        except Exception as e:
            logger.error(f"Light SER classification failed: {e}")
            return "neu"
