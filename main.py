from create_and_upload_to_s3 import create_bucket, upload_file_to_bucket
from generate_transcript import transcribe_audio
from generate_diarization_details import diarize_speakers_item_by_item
from generate_summary import generate_summary_for_transcript
import boto3
import os
import time

def process_audio_file(file_path):
    # Define AWS S3 bucket name
    bucket_name = 'sorrybuckettcb'  
    aws_region = 'us-east-1' 
    user_arn = "arn:aws:iam::381491989798:user/ekjotxo"

    # Assuming the bucket is already created and the file_path is valid
    print("Uploading audio file to S3 bucket...")
    upload_file_to_bucket(bucket_name, file_path)
    print("===============================================================================")    

    # Transcribe the uploaded audio file
    file_name = os.path.basename(file_path)
    print("Transcribing audio file...")
    local_file_path, transcript = transcribe_audio(bucket_name, file_name, aws_region=aws_region)
    print("===============================================================================")    

    # Diarize speakers from the transcription result
    print("Diarizing speakers: ")
    diarization_details = diarize_speakers_item_by_item(local_file_path)
    print("\n ===============================================================================")    
    
    # Generate summaries
    print("Generating summary details: ")
    summary = generate_summary_for_transcript(transcript)
    print("===============================================================================")    

    return {
        'transcript': transcript,
        'diarization_details': diarization_details,
        'summary': summary
    }
