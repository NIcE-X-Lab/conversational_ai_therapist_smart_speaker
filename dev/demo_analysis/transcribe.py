"""Transcribe the demo audio using faster-whisper (base.en model cached locally)."""
from faster_whisper import WhisperModel
from pathlib import Path

WAV = Path("/home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/dev/demo_analysis/audio/demo.wav")
OUT_TXT = Path("/home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/dev/demo_analysis/transcript.txt")
OUT_SRT = Path("/home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/dev/demo_analysis/transcript.srt")


def fmt_ts(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def main():
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(WAV),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    txt_lines = []
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        line = f"[{seg.start:7.2f} -> {seg.end:7.2f}] {seg.text.strip()}"
        print(line)
        txt_lines.append(line)
        srt_lines.append(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{seg.text.strip()}\n")
    OUT_TXT.write_text("\n".join(txt_lines))
    OUT_SRT.write_text("\n".join(srt_lines))
    print(f"\nsaved: {OUT_TXT} and {OUT_SRT}")


if __name__ == "__main__":
    main()
