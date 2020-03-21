import os
from random import shuffle
from shutil import unpack_archive, rmtree

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import train_test_split

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GCP Playground-34c3d1faef3b.json"
BUCKET_NAME = "citric-inkwell-268501"


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


def split_and_scale(data, test_size=0.2):
    """
    Splits and scales a dataset into testing and training sets.

    :param test_size: percent of data to be withheld for testing
    :param data: data to be split
    :return: split and scaled arrays of training and testing data
    """

    # Split data into labels and images
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])

    # Split data into test and train
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=42)

    # Scale images so that values are all between 0 and 1
    train_x_scaled = train_x.astype('float32')
    test_x_scaled = test_x.astype('float32')
    train_x_scaled /= 255
    test_x_scaled /= 255

    return train_x_scaled, test_x_scaled, train_y, test_y


def augment_images(augmentation_generator, data):
    """
    Augments images files.
    :param augmentation_generator: Augmentor Object
    :param data: list of image-like objects to be augmented
    :return: list of image-like objects
    """
    augmented_images = []
    for image in data:
        image = expand_dims(image, 0)
        augmentator = augmentation_generator.flow(image, batch_size=1)
        for i in range(12):
            batch = augmentator.next()
            augmented_images.append(batch[0])
    return augmented_images


def augment_and_upload(augmentation_generator, data, train_x, train_y):
    """
    Augment images and upload to cloud storage bucket.
    :param augmentation_generator: Generator Object
    :param data: data to be augmented
    :param train_x: training data to append augmentation to
    :param train_y: training labels to append augmentation to
    :return: null
    """
    # Package augmented images as training sets and send to cloud
    augmentor_cache = "./model_cache/augemntor_cache"
    if os.path.isdir("./model_cache/augemntor_cache"):
        print("Augmentor cache exists")
    else:
        os.mkdir(augmentor_cache)
        print("Augmentor cache created")

    name, IGD = augmentation_generator
    print("\n{} Beginning".format(str(name)))

    # Perform augmentation
    augmented_images = augment_images(IGD, data)
    train_x_aug = np.concatenate((train_x, np.array(augmented_images)))
    train_y_aug = np.concatenate((train_y, np.array([1 for i in range(len(augmented_images))])))
    local_train_x_path = os.path.join(augmentor_cache, "train_x_aug.npy")
    local_train_y_path = os.path.join(augmentor_cache, "train_y_aug.npy")

    # Save Augmented images
    np.save(local_train_x_path, train_x_aug)
    np.save(local_train_y_path, train_y_aug)
    cloud_train_x_aug_path = "training_sets/{}/{}_train_x_aug.npy".format(str(name), str(name))
    cloud_train_y_aug_path = "training_sets/{}/{}_train_y_aug.npy".format(str(name), str(name))
    print("     {} Finished augmenting".format(str(name)))
    print("     {} Beginning upload".format(str(name)))
    upload_blob(BUCKET_NAME, local_train_x_path, cloud_train_x_aug_path)
    upload_blob(BUCKET_NAME, local_train_y_path, cloud_train_y_aug_path)
    print("     {} finished uploading\n".format(str(name)))


def load_fire_dataset():
    """
    Loads dataset from cloud or local storage to memory
    :return: null
    """

    # Check for existence of local model_cache and create if it does not exist
    if os.path.isdir('./model_cache'):
        print("Model Cache Exists")
    else:
        os.mkdir("./model_cache")
        print("Created Model Cache")

    # Get list of existing files in bucket
    bucket_files = list_blobs(BUCKET_NAME)

    image_zip_name = "final_sorted_images.zip"
    image_zip_local_path = os.path.join('./model_cache/', image_zip_name)

    # Check if final_sorted_images.zip exists, and download if not
    if image_zip_name in bucket_files:
        print("final_sorted_images.zip found on cloud")
        if os.path.isfile(image_zip_local_path):
            print("final_sorted_images.zip already downloaded")
            pass
        else:
            download_blob(BUCKET_NAME, image_zip_name, image_zip_local_path)
            print("final_sorted_images.zip downloaded")

    # Unzip final_sorted_images.zip
    if os.path.isdir("./model_cache/sorted_images/"):
        print("Images unzipped")
        pass
    else:
        unpack_archive(image_zip_local_path, os.path.dirname("./model_cache/"))
        try:
            rmtree("./model_cache/__MACOSX")
        except OSError:
            print("Images unzipped cleanly")
        else:
            print("Images unzipped and directory cleaned")

    # Collect images and load into memory
    fire_image_dir = "./model_cache/sorted_images/fire"
    normal_image_dir = "./model_cache/sorted_images/selected_normal"

    training_data = resize_and_shuffle(fire_image_dir, 1)
    print("Fire images loaded")
    training_data += resize_and_shuffle(normal_image_dir, 0)
    print("Normal images loaded")

    # Save dataset to file
    np.save('./model_cache/unaugmented_training_data.npy', training_data)
    print("Unaugmented data saved")

    # split data into train and test
    train_x, test_x, train_y, test_y = split_and_scale(training_data)

    # Save and upload test data
    np.save('./model_cache/test_x.npy', test_x)
    np.save('./model_cache/test_y.npy', test_y)
    if 'test_set/test_x.npy' in bucket_files:
        print("test_x.npy already uploaded")
        pass
    else:
        upload_blob(BUCKET_NAME, './model_cache/test_x.npy', 'test_set/test_x.npy')
        print("test_x.npy successfully uploaded")
    if 'test_set/test_y.npy' in bucket_files:
        print("test_y.npy already uploaded")
        pass
    else:
        upload_blob(BUCKET_NAME, './model_cache/test_y.npy', 'test_set/test_y.npy')
        print("test_y.npy successfully uploaded")

    # Extract only images of fire to augment
    image_df = pd.DataFrame(training_data)
    fire_image_data = image_df[image_df[1] == 1]
    fire_images = fire_image_data[0].tolist()
    fire_labels = fire_image_data[1].tolist()

    # Define image augmentation generators

    augmentors = {
        'zoom_augmentation': ImageDataGenerator(rescale=1. / 255, zoom_range=0.3),
        'rotation_augmentation': ImageDataGenerator(rescale=1. / 255, rotation_range=180),
        'widthShift_augmentation': ImageDataGenerator(rescale=1. / 255, width_shift_range=0.3),
        'heightShift_augmentation': ImageDataGenerator(rescale=1. / 255, height_shift_range=0.3),
        'shear_augmentation': ImageDataGenerator(rescale=1. / 255, shear_range=0.3),
        'horizaontalFlip_augmentation': ImageDataGenerator(rescale=1. / 255, horizontal_flip=True),
        'verticalFlip_augmentation': ImageDataGenerator(rescale=1. / 255, vertical_flip=True),
        'full_augmentation': ImageDataGenerator(rescale=1. / 255,
                                                zoom_range=0.3,
                                                rotation_range=180,
                                                width_shift_range=0.3,
                                                height_shift_range=0.3,
                                                shear_range=0.3,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                fill_mode='nearest')
    }

    print('Augmenting images...')
    for augmentor in augmentors.items():
        augment_and_upload(augmentor, fire_images, train_x, train_y)

    print("Done")
