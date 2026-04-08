import numpy as np
import scipy.io.wavfile as wav

sample_rate = 16000
duration = 120 # 2 minutes
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Make soft brown noise
noise = np.random.normal(0, 1, len(t))
brown_noise = np.cumsum(noise)
# Normalize
brown_noise = brown_noise / np.max(np.abs(brown_noise)) * 0.05

# Add some soft slow chords (C Major 7: C, E, G, B)
f1, f2, f3, f4 = 261.63, 329.63, 392.00, 493.88
chord = (np.sin(2 * np.pi * f1 * t) + 
         np.sin(2 * np.pi * f2 * t) + 
         np.sin(2 * np.pi * f3 * t) + 
         np.sin(2 * np.pi * f4 * t))

# Soft continuous envelope (slow 8 second pulsing)
pulse = (np.sin(2 * np.pi * (1/8) * t) + 1) / 2 
chord = chord * pulse * 0.1

# Low thrumming bass
bass = np.sin(2 * np.pi * 65.41 * t) * 0.1

audio = brown_noise + chord + bass
audio = audio / np.max(np.abs(audio)) * 0.3

# Smooth fade in and out to prevent popping
fade_len = int(sample_rate * 2) # 2 seconds
audio[:fade_len] = audio[:fade_len] * np.linspace(0, 1, fade_len)
audio[-fade_len:] = audio[-fade_len:] * np.linspace(1, 0, fade_len)

# Convert to 16-bit PCM
audio_pcm = np.int16(audio * 32767)

wav.write('waiting_music.wav', sample_rate, audio_pcm)
print("Generated waiting_music.wav dynamically built ambient loop.")
