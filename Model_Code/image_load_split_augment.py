import os
from random import shuffle
from zipfile import ZipFile

import cv2
import numpy as np
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - " \
                                               "Deep Learning/GCP Playground-34c3d1faef3b.json"


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


def resize_and_shuffle(image_directory, label=0):
    """
    Collects, labels, resizes and shuffles images.
    :param image_directory: Path to image directory
    :param label: Label for image class
    :return: List of image-label objects
    """
    data = []

    # Label convention: fire = 1 , normal = 0

    for image in os.listdir(image_directory):
        path = os.path.join(image_directory, image)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            data.append([np.array(image), label])
        else:
            print("image_passed: " + path)
            pass
        shuffle(data)
    return data


def load_fire_dataset():
    """
    Loads dataset from cloud or local storage to memory
    :return: null
    """
    bucket_name = "citric-inkwell-268501"

    # Check for existence of local model_cache and create if it does not exist
    if os.path.isdir('./model_cache'):
        print("Model Cache Exists")
    else:
        os.mkdir("./model_cache")
        print("Created Model Cache")

    # Get list of existing files in bucket
    bucket_files = list_blobs(bucket_name)

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
    if os.path.isdir("./model_cache/sorted_images/"):
        print("Images unzipped")
        pass
    else:
        with ZipFile(image_zip_local_path, 'r') as zipObj:
            zipObj.extractall(path="./model_cache/")

    # Collect images and load into memory
    fire_image_dir = "./model_cache/sorted_images/fire"
    normal_image_dir = "./model_cache/sorted_images/selected_normal"

    training_data = resize_and_shuffle(fire_image_dir, 1)
    print("Fire images loaded")
    training_data += resize_and_shuffle(normal_image_dir, 0)
    print("Normal images loaded")

    # Save dataset to file
    np.save('./model_cache/unaugmented_training_data.npy', training_data)
    print("Data saved")

    print("Done")
