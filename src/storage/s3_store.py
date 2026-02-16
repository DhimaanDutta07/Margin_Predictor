import boto3
from botocore.exceptions import ClientError

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test",
)

BUCKET = "ml-artifacts"


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)


def upload(local_path: str, key: str):
    ensure_bucket()
    s3.upload_file(local_path, BUCKET, key)
    return f"s3://{BUCKET}/{key}"


def download(key: str, local_path: str):
    ensure_bucket()
    s3.download_file(BUCKET, key, local_path)
    return local_path
