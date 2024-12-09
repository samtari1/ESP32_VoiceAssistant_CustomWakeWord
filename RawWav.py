import wave

# File path to the WAV file
file_path = "ok.wav"

# Read the WAV file
with open(file_path, 'rb') as f:
    # Read the entire file as raw data
    raw_data = f.read()

    # Print raw data as hexadecimal
    print("Raw Data (Hex):")
    print(" ".join(f"{byte:02X}" for byte in raw_data[:256]))  # Print the first 256 bytes for brevity

    # Optionally, print the ASCII representation of the raw data
    print("\nRaw Data (ASCII):")
    print("".join(chr(byte) if 32 <= byte <= 126 else '.' for byte in raw_data[:256]))

# Use wave module to get file properties (optional)
with wave.open(file_path, 'rb') as wav_file:
    # Get audio file parameters
    params = wav_file.getparams()
    print(f"\nWAV Parameters: {params}")

    # Read the first 10 frames (raw audio data)
    wav_file.rewind()  # Ensure we're at the beginning of the audio data
    audio_data = wav_file.readframes(10)
    print("\nFirst 10 Frames (Raw):")
    print(" ".join(f"{byte:02X}" for byte in audio_data))

