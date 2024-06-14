import boto3
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
    """
    Download a file from an S3 bucket to a local file path.

    :param bucket_name: The name of the S3 bucket.
    :param s3_file_key: The key of the file in the S3 bucket.
    :param local_file_path: The local path where the file will be saved.
    :return: True if file was downloaded successfully, else False.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket_name, s3_file_key, local_file_path)
        logging.info(f"File {s3_file_key} downloaded successfully from {bucket_name} to {local_file_path}.")
        return True
    except ClientError as e:
        # This will catch errors such as file not found or access denied.
        logging.error(f"Error downloading file: {e}")
    except NoCredentialsError:
        logging.error("No AWS credentials were found")
    except PartialCredentialsError:
        logging.error("Incomplete AWS credentials were found")
    return False
