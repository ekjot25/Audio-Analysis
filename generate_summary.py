import boto3

def generate_summary_for_transcript(transcript):
    """
    Generate a summary from a transcript using AWS Comprehend.

    :param transcript: The transcript text.
    :return: A dictionary containing the dominant language, sentiment, key phrases, and named entities.
    """
    # Initialize the Comprehend client
    comprehend = boto3.client('comprehend')
    
    text = transcript[:5000]  # Limit text to the first 5000 characters to meet API constraints

    try:
        # Detect the dominant language
        response = comprehend.detect_dominant_language(Text=text)
        dominant_language = response['Languages'][0]['LanguageCode']
        
        # Detect sentiment
        sentiment_response = comprehend.detect_sentiment(Text=text, LanguageCode=dominant_language)
        sentiment = sentiment_response['Sentiment']

        # Detect named entities
        entities_response = comprehend.detect_entities(Text=text, LanguageCode=dominant_language)
        entities = [{'Text': entity['Text'], 'Type': entity['Type']} for entity in entities_response['Entities']]
        
        # Detect key phrases
        key_phrases_response = comprehend.detect_key_phrases(Text=text, LanguageCode=dominant_language)
        key_phrases = [phrase['Text'] for phrase in key_phrases_response['KeyPhrases']]
        

        # Return a structured response with all information
        return {
            "dominant_language": dominant_language,
            "sentiment": sentiment,
            "entities": entities,
            "key_phrases": key_phrases
        }

    except Exception as e:
        print(f"Error processing text analysis: {e}")
        return {"error": str(e)}

# Example usage in a Flask route
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/analyze_transcript', methods=['POST'])
def analyze_transcript():
    transcript = request.json.get('transcript', '')
    if transcript:
        analysis_results = generate_summary_for_transcript(transcript)
        return jsonify(analysis_results)
    else:
        return jsonify({"error": "No transcript provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
