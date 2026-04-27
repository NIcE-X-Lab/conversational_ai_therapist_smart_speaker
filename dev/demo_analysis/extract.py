"""Extract frames (every N seconds) and audio track from the demo mp4 using PyAV."""
import av
import numpy as np
from pathlib import Path
from PIL import Image
import wave

SRC = Path("/home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/mexa_llmtherapist_demo.mp4")
OUT = Path("/home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/dev/demo_analysis")
FRAMES_DIR = OUT / "frames"
AUDIO_WAV = OUT / "audio" / "demo.wav"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_WAV.parent.mkdir(parents=True, exist_ok=True)

INTERVAL_S = 3.0
SCALE_W = 1600  # downscale for analysis, preserving readability of terminal text


def extract_frames():
    container = av.open(str(SRC))
    v = container.streams.video[0]
    v.thread_type = "AUTO"
    fps = float(v.average_rate)
    duration_s = float(v.duration * v.time_base)
    step_frames = max(1, int(round(fps * INTERVAL_S)))
    target_indices = set(range(0, v.frames or 99999, step_frames))
    print(f"video fps={fps} dur={duration_s:.2f}s step_frames={step_frames} targets={len(target_indices)}")
    saved = 0
    idx = 0
    for frame in container.decode(video=0):
        if idx in target_indices:
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            new_w = min(SCALE_W, w)
            new_h = int(h * new_w / w)
            pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
            t = idx / fps
            path = FRAMES_DIR / f"frame_{int(t):04d}s.jpg"
            pil.save(path, quality=85)
            saved += 1
        idx += 1
    container.close()
    print(f"saved {saved} frames")


def extract_audio():
    container = av.open(str(SRC))
    a = container.streams.audio[0]
    # Resample to 16kHz mono PCM for Whisper
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    samples = []
    for packet in container.demux(a):
        for frame in packet.decode():
            out = resampler.resample(frame)
            if out is None:
                continue
            frames = out if isinstance(out, list) else [out]
            for f in frames:
                arr = f.to_ndarray()
                samples.append(arr.flatten())
    container.close()
    pcm = np.concatenate(samples).astype(np.int16)
    with wave.open(str(AUDIO_WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    print(f"wrote {AUDIO_WAV} samples={len(pcm)} dur={len(pcm)/16000:.2f}s")


if __name__ == "__main__":
    extract_frames()
    extract_audio()
