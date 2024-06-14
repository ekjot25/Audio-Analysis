import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import logging
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def preprocess_audio(file_path):
    """Preprocesses an audio file by converting it to mono and setting the sample rate."""
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        return audio
    except CouldntDecodeError as e:
        raise Exception(f"Error processing audio file {file_path}: {str(e)}")

def load_audio(file_path, target_sr=None):
    """Load and preprocess audio using pydub and librosa."""
    try:
        # First, preprocess the audio to ensure consistency in format
        preprocessed_audio = preprocess_audio(file_path)
        # Export the preprocessed audio to a temporary file and load with librosa
        temp_file = "temp.wav"
        preprocessed_audio.export(temp_file, format="wav")
        audio, sr = librosa.load(temp_file, sr=target_sr)
        os.remove(temp_file)  # Clean up the temporary file
        if np.isnan(audio).any():
            logging.warning(f"Audio file {file_path} contains NaN values.")
            return None, None
        return audio, sr
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        return None, None


def spectral_gate(audio, sr, stride=0.01, threshold=20, min_silence_duration=0.05):
    # Convert to spectrogram
    stft = librosa.stft(audio)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Calculate noise profile
    silent_frames = librosa.effects.split(audio, top_db=threshold, frame_length=int(sr*stride), hop_length=int(sr*min_silence_duration))
    noise_profile = np.mean(np.abs(librosa.stft(audio[silent_frames[0][0]:silent_frames[0][1]])), axis=1)
    noise_profile = np.minimum(noise_profile, np.mean(magnitude, axis=1))

    # Spectral gating
    gain = np.maximum(magnitude - noise_profile[:, np.newaxis], 0) / (magnitude + 1e-10)
    magnitude_reduced = gain * magnitude

    # Convert back to time domain
    reduced_audio = librosa.istft(magnitude_reduced * np.exp(1j * phase))
    return reduced_audio

# Utility Functions
def butter_lowpass_filter(data, cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_level * noise
    return augmented_audio / np.max(np.abs(augmented_audio))

def change_pitch_and_speed(audio, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_factor)

def apply_time_stretching(audio, rate=0.8):
    # The 'rate' parameter is now a keyword argument
    return librosa.effects.time_stretch(y=audio, rate=rate)

def preprocess_and_extract_features(audio, sr, chunk_size=10, n_mfcc=20, include_chroma=True, include_contrast=True, return_feature_data=True):
    print(f"Sample rate: {sr}, Audio Length: {len(audio)}")
    num_chunks = len(audio) // (chunk_size * sr)
    print(f"Number of chunks: {num_chunks}")

    all_features = []
    feature_data = {'mfccs': [], 'chroma': [], 'contrast': []}  # Initialize feature data dictionary

    for i in range(num_chunks):
        chunk = audio[i * chunk_size * sr:(i + 1) * chunk_size * sr]
        filtered_chunk = butter_lowpass_filter(chunk, cutoff=2000, sr=sr)
        normalized_chunk = librosa.util.normalize(filtered_chunk)

        # Extract and process MFCCs
        mfccs = librosa.feature.mfcc(y=normalized_chunk, sr=sr, n_mfcc=n_mfcc)
        mfccs_scaled = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        features = mfccs_scaled.flatten()
        if return_feature_data:
            feature_data['mfccs'].append(mfccs_scaled)

        # Process Chroma and Spectral Contrast if included
        if include_chroma or include_contrast:
            chroma, contrast = None, None
            if include_chroma:
                chroma = librosa.feature.chroma_cqt(y=normalized_chunk, sr=sr)
                chroma_scaled = (chroma - np.mean(chroma)) / np.std(chroma)
                features = np.hstack((features, chroma_scaled.flatten()))
                if return_feature_data:
                    feature_data['chroma'].append(chroma_scaled)

            if include_contrast:
                contrast = librosa.feature.spectral_contrast(y=normalized_chunk, sr=sr)
                contrast_scaled = (contrast - np.mean(contrast)) / np.std(contrast)
                features = np.hstack((features, contrast_scaled.flatten()))
                if return_feature_data:
                    feature_data['contrast'].append(contrast_scaled)

        # Check for NaNs
        if np.isnan(features).any():
            logging.warning("NaN values detected after feature extraction.")
            continue  # Skip this chunk

        all_features.append(features)

    all_features = np.array(all_features)
    if np.isnan(all_features).any():
        logging.warning("NaN values detected in all features.")
    print(f"All features shape: {all_features.shape}")

    if return_feature_data:
        return all_features, feature_data
    else:
        return all_features

def process_audio_files(folder_path, augment=True, return_feature_data=True):
    """Process multiple audio files from a folder."""
    all_features = []
    all_feature_data = []  # To store feature data for visualization

    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            audio, sr = load_audio(os.path.join(folder_path, file))
            if audio is not None:
                # Proceed with further processing like denoising and feature extraction
                # Example: audio = spectral_gate(audio, sr)
                pass
            else:
                logging.warning(f"Skipping file {file} due to loading error.")
   

def process_audio_file(file_path, target_sr=None):
    """
    Process a single audio file to apply filtering and denoising, then return the processed audio waveform.

    Args:
    - file_path (str): Path to the audio file.
    - target_sr (int): Target sample rate.

    Returns:
    - processed_audio (np.array): The processed audio waveform.
    - sr (int): The sample rate of the processed audio.
    """
    try:
        # Load audio
        audio, sr = load_audio(file_path, target_sr=target_sr)
        if audio is None:
            logging.error("Failed to load audio.")
            return None, None

        # Apply spectral gate (denoising)
        audio = spectral_gate(audio, sr)

        # Apply low-pass filter
        audio = butter_lowpass_filter(audio, cutoff=2000, sr=sr)

        return audio, sr
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {e}")
        return None, None
