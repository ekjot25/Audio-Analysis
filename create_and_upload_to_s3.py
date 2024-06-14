import boto3
import os
from botocore.exceptions import ClientError
from werkzeug.utils import secure_filename
import json

def create_bucket(bucket_name, region='us-east-1'):
    """
    Create an S3 bucket in the specified region and return a success or error message.

    :param bucket_name: Name of the bucket to create.
    :param region: AWS region where the bucket will be created.
    :return: Tuple of success (bool) and message (str).
    """
    s3_client = boto3.client('s3', region_name=region)
    try:
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
        return True, f"Bucket {bucket_name} created successfully in {region}."
    except ClientError as e:
        return False, f"Error creating bucket: {e}"

def upload_file_to_bucket(bucket_name, file_path):
    """
    Upload a file to an S3 bucket and return a success or error message.

    :param bucket_name: Name of the bucket to upload the file to.
    :param file_path: Local path of the file to upload.
    :return: Tuple of success (bool) and message (str).
    """
    s3_client = boto3.client('s3')
    file_name = secure_filename(os.path.basename(file_path))
    try:
        s3_client.upload_file(file_path, bucket_name, file_name)
        return True, f"File {file_name} uploaded successfully to {bucket_name}."
    except ClientError as e:
        return False, f"Error uploading file: {e}"

def attach_bucket_policy(user_arn, bucket_name):
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "AWS": user_arn
                },
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}/*"
                ]
            }
        ]
    }

    # Convert the policy to JSON string
    policy_json = json.dumps(policy)

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Attach the policy to the bucket
    response = s3_client.put_bucket_policy(
        Bucket=bucket_name,
        Policy=policy_json
    )

    print("Bucket policy attached successfully.")

