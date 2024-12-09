import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm

class SpeechSpectrogramProcessor:
    def __init__(self, speech_data_dir='speech_data'):
        # Constants
        self.SPEECH_DATA = speech_data_dir
        self.EXPECTED_SAMPLES = 16000  # 1 second at 16 kHz
        self.NOISE_FLOOR = 0.1
        self.MINIMUM_VOICE_LENGTH = self.EXPECTED_SAMPLES // 4
        
        # Word categories
        self.words = [
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 
            'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 
            'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 
            'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 
            'wow', 'yes', 'zero', '_background'
        ]        

        # Data storage
        self.train = []
        self.validate = []
        self.test = []
        
        # Split ratios
        self.TRAIN_SIZE = 0.8
        self.VALIDATION_SIZE = 0.1
        self.TEST_SIZE = 0.1

    def get_files(self, word: str) -> List[str]:
        """Get all wav files for a given word."""
        word_path = os.path.join(self.SPEECH_DATA, word)
        return [os.path.join(word_path, f) for f in os.listdir(word_path) if f.endswith('.wav')]

    def get_voice_position(self, audio: np.ndarray, noise_floor: float) -> Tuple[int, int]:
        """Find the position of the voice in the audio."""
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        
        # Find where audio amplitude exceeds noise floor
        above_noise = np.abs(audio) > noise_floor
        voice_indices = np.where(above_noise)[0]
        
        if len(voice_indices) == 0:
            return 0, len(audio)
        
        return voice_indices[0], voice_indices[-1]

    def get_voice_length(self, audio: np.ndarray, noise_floor: float) -> int:
        """Calculate the length of the voice segment."""
        start, end = self.get_voice_position(audio, noise_floor)
        return end - start

    def is_voice_present(self, audio: np.ndarray, noise_floor: float, required_length: int) -> bool:
        """Check if sufficient voice is present in the audio."""
        voice_length = self.get_voice_length(audio, noise_floor)
        return voice_length >= required_length

    def is_valid_file(self, file_path: str) -> bool:
        """Validate an audio file."""
        try:
            audio, _ = librosa.load(file_path, sr=16000, duration=1.0)
            
            # Check length
            if len(audio) != self.EXPECTED_SAMPLES:
                return False
            
            # Check voice presence
            if not self.is_voice_present(audio, self.NOISE_FLOOR, self.MINIMUM_VOICE_LENGTH):
                return False
            
            return True
        except Exception:
            return False

    def get_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Generate spectrogram from audio."""
        # Normalize audio
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        
        # Create spectrogram using librosa
        spectrogram = librosa.stft(audio, n_fft=320, hop_length=160)
        spectrogram = np.abs(spectrogram)**2
        
        # Reduce frequency bins
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        return spectrogram

    def process_file(self, file_path: str) -> np.ndarray:
        """Process an audio file into a spectrogram."""
        # Load audio
        audio, _ = librosa.load(file_path, sr=16000, duration=1.0)
        
        # Normalize
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        
        # Reposition audio
        voice_start, voice_end = self.get_voice_position(audio, self.NOISE_FLOOR)
        end_gap = len(audio) - voice_end
        random_offset = int(np.random.uniform(0, voice_start + end_gap))
        audio = np.roll(audio, -random_offset + end_gap)
        
        # Add background noise
        background_volume = np.random.uniform(0, 0.1)
        background_files = self.get_files('_background_noise_')
        background_file = np.random.choice(background_files)
        background, _ = librosa.load(background_file, sr=16000, duration=1.0)
        background = background - np.mean(background)
        background = background / np.max(np.abs(background))
        
        # Mix audio with background
        audio = audio + background_volume * background
        
        # Generate spectrogram
        return self.get_spectrogram(audio)

    def process_word(self, word: str, repeat: int = 1):
        """Process files for a specific word."""
        label = self.words.index(word)
        
        # Get valid files
        file_names = [f for f in self.get_files(word) if self.is_valid_file(f)]
        np.random.shuffle(file_names)
        
        # Split into train, validate, test
        train_size = int(self.TRAIN_SIZE * len(file_names))
        validation_size = int(self.VALIDATION_SIZE * len(file_names))
        
        # Process and extend datasets
        def process_subset(subset, is_repeat=False):
            processed = []
            for file_name in tqdm(subset, desc=f"{word} {'(repeated)' if is_repeat else ''}"):
                processed.extend([(self.process_file(file_name), label) for _ in range(repeat)])
            return processed
        
        self.train.extend(process_subset(file_names[:train_size], repeat > 1))
        self.validate.extend(process_subset(
            file_names[train_size:train_size+validation_size], 
            repeat > 1
        ))
        self.test.extend(process_subset(file_names[train_size+validation_size:], repeat > 1))

    def process_background(self, file_path: str, label: int):
        """Process background noise files."""
        audio, _ = librosa.load(file_path, sr=16000)
        samples = []
        
        # Generate samples from background noise
        for start in range(0, len(audio) - self.EXPECTED_SAMPLES, 8000):
            section = audio[start:start + self.EXPECTED_SAMPLES]
            spectrogram = self.get_spectrogram(section)
            samples.append((spectrogram, label))
        
        # Simulate random utterances
        for _ in range(1000):
            start = np.random.randint(0, len(audio) - self.EXPECTED_SAMPLES)
            section = audio[start:start + self.EXPECTED_SAMPLES]
            
            # Create pseudo voice section
            voice_length = int(np.random.uniform(self.MINIMUM_VOICE_LENGTH//2, self.EXPECTED_SAMPLES))
            voice_start = np.random.randint(0, self.EXPECTED_SAMPLES - voice_length)
            
            hamming = np.hamming(voice_length)
            result = np.zeros_like(section)
            result[voice_start:voice_start+voice_length] = hamming * section[voice_start:voice_start+voice_length]
            
            spectrogram = self.get_spectrogram(result)
            samples.append((spectrogram, label))
        
        # Split and add to datasets
        np.random.shuffle(samples)
        train_size = int(self.TRAIN_SIZE * len(samples))
        validation_size = int(self.VALIDATION_SIZE * len(samples))
        
        self.train.extend(samples[:train_size])
        self.validate.extend(samples[train_size:train_size+validation_size])
        self.test.extend(samples[train_size+validation_size:])

    def process_all_data(self):
        """Process all words and background noise."""
        for word in tqdm(self.words, desc="Processing words"):
            if '_' not in word:
                # Add more examples of marvin to balance training set
                repeat = 50 if word == 'marvin' else 1
                self.process_word(word, repeat=repeat)
        
        # Process background noise files
        for file_path in tqdm(self.get_files('_background_noise_'), desc="Processing Background Noise"):
            self.process_background(file_path, self.words.index("_background"))
        
        # Shuffle training data
        np.random.shuffle(self.train)
        
        return self

    def save_data(self, prefix=''):
        """Save processed data to compressed numpy files."""
        X_train, Y_train = zip(*self.train)
        X_validate, Y_validate = zip(*self.validate)
        X_test, Y_test = zip(*self.test)
        
        np.savez_compressed(f"{prefix}training_spectrogram.npz", X=X_train, Y=Y_train)
        np.savez_compressed(f"{prefix}validation_spectrogram.npz", X=X_validate, Y=Y_validate)
        np.savez_compressed(f"{prefix}test_spectrogram.npz", X=X_test, Y=Y_test)
        
        return self

    def plot_spectrograms(self, word: str, n_images: int = 20):
        """Plot spectrograms for a specific word."""
        # Find indices for the specified word
        word_index = self.words.index(word)
        X_word = np.array([x for x, y in self.train if y == word_index])
        
        # Plot images
        if len(X_word) > 0:
            fig, axes = plt.subplots(5, 4, figsize=(12, 15))
            axes = axes.flatten()
            for img, ax in zip(X_word[:n_images], axes):
                ax.imshow(img, cmap='viridis', origin='lower')
                ax.axis("off")
            plt.tight_layout()
            
            # Generate a unique file name for the plot
            file_name = f"plot_{word}_images.png"
            plt.savefig(file_name)  # Save the plot with the unique file name
            plt.show()
            print(f"Plot saved as {file_name}")
        else:
            print(f"No images found for word: {word}")

# Example usage
if __name__ == "__main__":
    processor = SpeechSpectrogramProcessor()
    processor.process_all_data().save_data()
    
    # Visualize spectrograms for some words
    # processor.plot_spectrograms('marvin')
    # processor.plot_spectrograms('_background')