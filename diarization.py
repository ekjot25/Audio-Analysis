import logging
from pyannote.audio import Pipeline
import numpy as np

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def perform_diarization(audio_path, token, sr=None):
    """Performs speaker diarization on a given audio file using Pyannote's pretrained model.

    Args:
        audio_path (str): Path to the audio file.
        token (str): Hugging Face authentication token.
        sr (int, optional): Sample rate of the audio. If None, uses default from audio file.

    Returns:
        Tuple containing the diarization result and speaker labels array, or (None, None) if an error occurs.
    """
    try:
        # Initialize the diarization pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
        diarization = pipeline(audio_path)

        # Handle dynamic sample rate
        if sr is None:
            from librosa.core import get_samplerate
            sr = get_samplerate(audio_path)

        # Process diarization results
        intervals = [(segment.start, segment.end, label) for segment, _, label in diarization.itertracks(yield_label=True)]
        unique_speakers = sorted(set(label for _, _, label in intervals))
        speaker_to_int = {speaker_id: i for i, speaker_id in enumerate(unique_speakers)}

        # Update speaker IDs in intervals to integers
        intervals = [(start, end, speaker_to_int[label]) for start, end, label in intervals]

        # Creating sample-wise speaker labels array
        duration = diarization.get_timeline().duration()
        num_samples = int(duration * sr)
        speaker_labels = np.zeros(num_samples)

        for start_time, end_time, speaker_id in intervals:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            speaker_labels[start_sample:end_sample] = speaker_id

        return diarization, speaker_labels

    except Exception as e:
        logging.error(f"Error during diarization of file {audio_path}: {e}")
        return None, None
