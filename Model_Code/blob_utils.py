import os

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GCP Playground-34c3d1faef3b.json"


def list_blobs(bucket_name):
    """
    List all blobs in cloud storage bucket
    :param bucket_name: Name of cloud storage bucket
    :return blobs: list of blobs in cloud storage bucket
    """
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        print(blob.name)

    return blobs


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Download a blob to file-like object from cloud storage bucket.
    :param bucket_name: Name of cloud storage bucket
    :param source_blob_name: Path to blob in cloud storage bucket
    :param destination_file_name: Name of downloaded file
    :return: null
    """
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file-like blob to a cloud storage bucket
    :param bucket_name: Name of bucket to upload to
    :param source_file_name: Name of file to upload
    :param destination_blob_name: Path to destination in storage bucket
    :return: null
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
