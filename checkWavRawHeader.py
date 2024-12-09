import wave

# File path to the WAV file
file_path = "0a2b400e_nohash_0.wav"

# Extract and print the raw header
with open(file_path, 'rb') as file:
    # Read the first 44 bytes (standard WAV header size)
    raw_header = file.read(44)
    # Print raw header in hexadecimal format
    print("Raw Header (Hex):")
    print(" ".join(f"{byte:02X}" for byte in raw_header))
    # Optionally, print the raw header as ASCII where possible
    print("\nRaw Header (ASCII):")
    print("".join(chr(byte) if 32 <= byte <= 126 else '.' for byte in raw_header))

# Use the wave module to extract properties for additional context
with wave.open(file_path, 'rb') as wav_file:
    channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()  # Bytes per sample
    sample_rate = wav_file.getframerate()
    bit_depth = sample_width * 8  # Convert bytes to bits

    print(f"\nChannels: {channels}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Bit Depth: {bit_depth} bits")

