import cv2
import numpy as np
import os
from random import shuffle
from zipfile import ZipFile
from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/GCP Playground-34c3d1faef3b.json"
storage_client = storage.Client()

bucket_name = "citric-inkwell-268501"

def list_blobs(bucket_name):
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        print(blob.name)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

# Check for existance of local model_cache and create if it does not exist
if os.path.isdir('./model_cache'):
    print("Model Cache Exists")
else:
    os.mkdir("./model_cache")
    print("Created Model Cache")

# Get list of existing files in bucket
bucket_files = [blob.name for blob in storage_client.list_blobs(bucket_name)]

image_zip_name = "final_sorted_images.zip"
image_zip_local_path = os.path.join('./model_cache/', image_zip_name)

# Check if final_sorted_images.zip exists, and download if not
if image_zip_name in bucket_files:
    print("final_sorted_images.zip found on cloud")
    if os.path.isfile(image_zip_local_path):
        print("final_sorted_images.zip already downloaded")
        pass
    else:
        download_blob(bucket_name, image_zip_name, image_zip_local_path)
        print("final_sorted_images.zip downloaded")

# Unzip final_sorted_images.zip
with ZipFile(image_zip_local_path, 'r') as zipObj:
    zipObj.extractall()


print("done")