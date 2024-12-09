import wave

# Open the WAV file
file_path = "0a2b400e_nohash_0.wav"
with wave.open(file_path, 'rb') as wav_file:
    # Extract properties
    channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()  # Bytes per sample
    sample_rate = wav_file.getframerate()
    bit_depth = sample_width * 8  # Convert bytes to bits
    
    print(f"Channels: {channels}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Bit Depth: {bit_depth} bits")

