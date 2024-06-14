
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

def extract_features(audio):
    """Extracts MFCC and its delta features from an audio segment, adding more complexity to feature extraction.
    
    Parameters:
    - audio: AudioSegment, the audio segment from which to extract features.
    
    Returns:
    - features: ndarray, the extracted features as a numpy array.
    """
    try:
        mfccs = librosa.feature.mfcc(np.array(audio.get_array_of_samples()), sr=audio.frame_rate, n_mfcc=20)
        delta = librosa.feature.delta(mfccs, order=1)
        delta_delta = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta, delta_delta])
        return features.T
    except Exception as e:
        raise RuntimeError(f"Failed to extract features: {str(e)}")

def cluster_audio(features):
    """Clusters the audio features using KMeans, adding error handling and complexity.
    
    Parameters:
    - features: ndarray, the features of the audio segments.
    
    Returns:
    - clusters: ndarray, the cluster labels for each feature vector.
    
    Raises:
    - RuntimeError: if clustering fails.
    """
    try:
        kmeans = KMeans(n_clusters=3, random_state=42)  # More clusters for added complexity
        clusters = kmeans.fit_predict(features)
        return clusters
    except NotFittedError as e:
        raise RuntimeError(f"Clustering failed: {str(e)}")
