import sys, pyaudio
def run_check():
    # 1. Venv Check
    is_venv = sys.base_prefix != sys.prefix
    print(f"{'✅' if is_venv else '❌'} VENV ACTIVE")
    
    # 2. Audio Hardware Check
    p = pyaudio.PyAudio()
    in_devs = [p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0]
    out_devs = [p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxOutputChannels'] > 0]
    
    print(f"{'✅' if in_devs else '❌'} MICS FOUND: {in_devs}")
    print(f"{'✅' if out_devs else '❌'} SPEAKERS FOUND: {out_devs}")

    # 3. Loopback Test (3 seconds)
    if in_devs and out_devs:
        print("🎤 RECORDING (3s)... Talk now!")
        s_in = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = [s_in.read(1024) for _ in range(0, int(44100 / 1024 * 3))]
        print("🔊 PLAYING BACK...")
        s_out = p.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True, frames_per_buffer=1024)
        for d in frames: s_out.write(d)
        print("✅ TEST COMPLETE")
    p.terminate()

if __name__ == "__main__":
    run_check()
