
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def generate_summary(audio_text):
    """Generates a summary for given audio text, with improved error handling and complexity.
    
    Parameters:
    - audio_text: str, the text to summarize.
    
    Returns:
    - summary: str, the generated summary.
    
    Raises:
    - ValueError: if input text is empty.
    """
    if not audio_text.strip():
        raise ValueError("Input text for summarization is empty.")
    
    try:
        sentences = sent_tokenize(audio_text)
        stop_words = set(stopwords.words('english') + list('.,!?'))
        filtered_sentences = [' '.join([w for w in word_tokenize(sentence) if not w.lower() in stop_words]) for sentence in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_sentences)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        scores = np.sum(cosine_sim, axis=1)
        top_sentences_idx = scores.argsort()[-3:][::-1]  # Taking top 3 sentences
        summary = ' '.join([filtered_sentences[i] for i in top_sentences_idx])
        return summary
    except Exception as e:
        raise RuntimeError(f"Failed to generate summary: {str(e)}")

