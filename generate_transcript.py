from download_from_s3 import download_file_from_s3
import urllib.request
import json
import boto3
import uuid

def transcribe_audio(bucket_name, file_name):
    """
    Transcribe an audio file stored in an S3 bucket using AWS Transcribe.

    :param bucket_name: The name of the S3 bucket where the audio file is stored.
    :param file_name: The name of the audio file to transcribe.
    :return: The transcription text if the job is completed successfully, otherwise None.
    """
    transcribe = boto3.client('transcribe')
    job_name = str(uuid.uuid4())  # Generate a unique job name
    job_uri = f's3://{bucket_name}/{file_name}'

    # Start transcription job
    print("Starting transcription job for the audio file.")
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat=file_name.split('.')[-1],
        LanguageCode='en-US',
        OutputBucketName= bucket_name,
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 10
        }
        
    )

    # Wait for transcription job to complete
    print("Waiting for transcription job to complete.")
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break

    print("Transcription job status: " + status['TranscriptionJob']['TranscriptionJobStatus'])
    
    s3_file_key = job_name + '.json'
    local_file_path = '/Users/ekjot/Downloads/AWS 2/' + s3_file_key  
    download_file_from_s3(bucket_name, s3_file_key, local_file_path)

    # Get transcription results
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        print("Retrieving transcript.")
        transcription_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        try:
            with open(local_file_path, 'r') as file:
                transcription_result = json.load(file)
        
            transcription_text = transcription_result['results']['transcripts'][0]['transcript']
            print(transcription_text)
        except Exception as e:
            print(f"Error reading transcription from local file: {e}")

    else:
        print("Transcription job failed.")
        return None
    return s3_file_key, local_file_path, transcription_text

# Example usage in a Flask route
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/transcribe_audio', methods=['POST'])
def handle_transcription():
    try:
        bucket_name = request.form['bucket_name']
        file_name = request.form['file_name']
        job_name, transcription_text = transcribe_audio(bucket_name, file_name)
        return jsonify({'job_name': job_name, 'transcription_text': transcription_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
