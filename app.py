import boto3
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import os
import time
import json
from werkzeug.utils import secure_filename
from generate_transcript import transcribe_audio
from generate_summary import generate_summary_for_transcript
from generate_diarization_details import diarize_speakers_by_punctuation
from create_and_upload_to_s3 import upload_file_to_bucket
from download_from_s3 import download_file_from_s3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}
app.secret_key = 'OoMUkAI70pnemOslx0eyLfTuV7oJLFMid1dXxVZ4'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file', None)
        if not file or file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            local_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(local_file_path)  # Save the file to the local filesystem

            # Upload the file to AWS S3
            bucket_name = 'sorrybuckettcb'  # Ensure this bucket exists in your AWS account
            upload_file_to_bucket(bucket_name, local_file_path)

            # Transcribe the uploaded audio file
            print("Transcribing audio file...")
            s3_file_key, transcript_path, transcript_text = transcribe_audio(bucket_name, filename)
            
		            
            # Download transcription JSON from S3
            local_transcript_path = os.path.join('/Users/ekjot/Downloads/AWS 2', transcript_path)
            print(s3_file_key, transcript_path,local_transcript_path)
            if download_file_from_s3(bucket_name, s3_file_key, local_transcript_path):
                print("Transcription file downloaded successfully.")

                # Diarize the transcription result
                print("Diarizing speakers...")
                with open(local_transcript_path, 'r') as file:
                    transcription_data = json.load(file)
                diarization_details = diarize_speakers_by_punctuation(transcription_data)

                # Generate summary using AWS Comprehend
                print("Generating summary...")
                summary_results = generate_summary_for_transcript(transcript_text)

                # Display results
                return render_template('results.html', summary=summary_results, transcript=transcript_text, diarization=diarization_details)
            else:
                flash("Failed to download the transcription file.")
                return redirect(request.url)

    return render_template('upload.html')

@app.route('/results')
def show_results():
    # This route can display results stored from the upload_file view
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)