import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from typing import List, Tuple


class AudioProcessor:
    def __init__(self, speech_data_path: str, noise_floor: float):
        self.SPEECH_DATA = speech_data_path
        self.NOISE_FLOOR = noise_floor

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize the audio to have zero mean and unit max absolute value."""
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        return audio

    def get_voice_position(self, audio: np.ndarray) -> Tuple[int, int]:
        """Find the position of the voice in the audio."""
        above_noise = np.abs(audio) > self.NOISE_FLOOR
        voice_indices = np.where(above_noise)[0]
        
        if len(voice_indices) == 0:
            return 0, len(audio)
        
        return voice_indices[0], voice_indices[-1]

    def reposition_audio(self, audio: np.ndarray) -> np.ndarray:
        """Reposition audio by rolling based on voice position."""
        voice_start, voice_end = self.get_voice_position(audio)
        end_gap = len(audio) - voice_end
        return np.roll(audio, -voice_start + end_gap)

    def get_files(self, word: str) -> List[str]:
        """Get all wav files for a given word."""
        word_path = os.path.join(self.SPEECH_DATA, word)
        return [os.path.join(word_path, f) for f in os.listdir(word_path) if f.endswith('.wav')]

    def add_background_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add random background noise to the audio."""
        background_volume = np.random.uniform(0, 0.1)
        background_files = self.get_files('_background_noise_')
        background_file = np.random.choice(background_files)
        background, _ = librosa.load(background_file, sr=16000, duration=1.0)
        background = self.normalize_audio(background)
        if len(audio) < len(background):
            background = background[:len(audio)]
        elif len(audio) > len(background):
            padding = len(audio) - len(background)
            background = np.pad(background, (0, padding), mode='constant')
        return audio + background_volume * background

    def get_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Generate spectrogram from audio."""
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        spectrogram = librosa.stft(audio, n_fft=320, hop_length=160)
        spectrogram = np.abs(spectrogram)**2
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram


def process_and_plot_audio(folder_path: str, speech_data_path: str, num_files: int = 10, noise_floor: float = 0.02):
    processor = AudioProcessor(speech_data_path, noise_floor)
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    selected_files = wav_files[:num_files]

    plt.figure(figsize=(15, num_files * 5))

    for idx, file_name in enumerate(selected_files):
        file_path = os.path.join(folder_path, file_name)
        audio, sr = librosa.load(file_path, sr=16000, duration=1.0)

        normalized_audio = processor.normalize_audio(audio)
        repositioned_audio = processor.reposition_audio(normalized_audio)
        noisy_audio = processor.add_background_noise(repositioned_audio)
        spectrogram = processor.get_spectrogram(noisy_audio)

        time = np.linspace(0, len(normalized_audio) / sr, len(normalized_audio))

        plt.subplot(num_files, 4, idx * 4 + 1)
        plt.plot(time, normalized_audio)
        plt.title(f"Original (Normalized): {file_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(num_files, 4, idx * 4 + 2)
        plt.plot(time, repositioned_audio)
        plt.title("Repositioned")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(num_files, 4, idx * 4 + 3)
        plt.plot(time, noisy_audio)
        plt.title("With Noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(num_files, 4, idx * 4 + 4)
        librosa.display.specshow(spectrogram, sr=sr, hop_length=160, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()


# Example usage
folder_path = "speech_data/marvin/"
speech_data_path = "speech_data/"
process_and_plot_audio(folder_path, speech_data_path, num_files=10, noise_floor=0.02)
