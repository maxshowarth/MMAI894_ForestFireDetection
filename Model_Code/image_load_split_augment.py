import cv2
import numpy as np
from numpy import expand_dims
import os
from random import shuffle
from zipfile import ZipFile
from google.cloud import storage
import os
from keras.preprocessing.image import ImageDataGenerator
from shutil import unpack_archive, rmtree
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk




os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/GCP Playground-34c3d1faef3b.json"
storage_client = storage.Client()

bucket_name = "citric-inkwell-268501"

# Utility function definitions
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

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def showNumpyImage(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.show()

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

unaugmented_training_data = []

# Label convention: fire = 1 , normal = 0

for image in os.listdir(fire_image_dir):
    label = 1
    path = os.path.join(fire_image_dir, image)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.resize(image, (224, 224))
        unaugmented_training_data.append([np.array(image), label])
    else:
        pass
    shuffle(unaugmented_training_data)
print("Fire images loaded")

for image in os.listdir(normal_image_dir):
    label = 0
    path = os.path.join(normal_image_dir, image)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.resize(image, (224, 224))
        unaugmented_training_data.append([np.array(image), label])
    else:
        "image_passed"
        pass
    shuffle(unaugmented_training_data)
print("Normal images loaded")
np.save('./model_cache/unaugmented_training_data.npy', unaugmented_training_data)
print("Unaugmented data saved")


# Split data into labels and images
x = np.array([i[0] for i in unaugmented_training_data])
y = np.array([i[1] for i in unaugmented_training_data])

# Split data into test and train
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Scale images so that values are all between 0 and 1
train_x_scaled = train_x.astype('float32')
test_x_scaled = test_x.astype('float32')
train_x_scaled /= 255
test_x_scaled /= 255


# Begin Image Augmentation

# Extract only images of fire to augment
image_df = pd.DataFrame(unaugmented_training_data)
fire_image_data = image_df[image_df[1]==1]
fire_images = fire_image_data[0].tolist()
fire_labels = fire_image_data[1].tolist()

# Define image augmentation generators

zoom_augmentation = ImageDataGenerator(rescale=1./255, zoom_range=0.3)
rotation_augmentation = ImageDataGenerator(rescale=1./255, rotation_range=180)
widthShift_augmentation = ImageDataGenerator(rescale=1./255, width_shift_range=0.3)
heightShift_augmentation = ImageDataGenerator(rescale=1./255, height_shift_range=0.3)
shear_augmentation = ImageDataGenerator(rescale=1./255, shear_range=0.3)
horizaontalFlip_augmentation = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
verticalFlip_augmentation = ImageDataGenerator(rescale=1./255, vertical_flip=True)
full_augmentation = ImageDataGenerator(rescale=1./255,
                                       zoom_range=0.3,
                                       rotation_range=180,
                                   width_shift_range=0.3,
                                       height_shift_range=0.3,
                                       shear_range=0.3,
                                   horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')

# Define augmentation function
def augmentImages(augmentation_generator):
    augmented_images = []
    for image in fire_images:
        image = expand_dims(image, 0)
        augmentator = augmentation_generator.flow(image, batch_size=1)
        for i in range(12):
            batch = augmentator.next()
            augmented_images.append(batch[0])
    return(augmented_images)

# augmentation_generators = [zoom_augmentation, rotation_augmentation, widthShift_augmentation]
# augmentedDataPack = []

# Do augmentation
zoomed_images = augmentImages(zoom_augmentation)
rotation_images = augmentImages(rotation_augmentation)
widthShift_images = augmentImages(widthShift_augmentation)
heightShift_images = augmentImages(heightShift_augmentation)
shear_images = augmentImages(shear_augmentation)
horizontalFlip_images = augmentImages(horizaontalFlip_augmentation)
verticalFlip_images = augmentImages(verticalFlip_augmentation)
fullyAugmented_images = augmentImages(full_augmentation)
# print("done")
# showNumpyImage(fullyAugmented_images[1])

# Package augmented images as training sets and send to cloud
def packageAndUpload(augmentedImages, augmentation_name):
    train_x_aug = np.concatenate((train_x_scaled), np.array(augmentedImages))
    aug_labels = []
    for i in range(len(augmentImages())):
        aug_labels.append(1)
    train_y_aug = np.concatenate((train_y, np.array(aug_labels)))
    training_pack = [train_x_aug, train_y_aug]
    filePath = os.path.join("./model_cache/", "{}.pkl".format(str(augmentation_name)))
    f = open(filePath, 'wb')
    pk.dump(training_pack, f)
    f.close()
    upload_blob(bucket_name,filePath, augmentation_name)

packageAndUpload(zoomed_images, "zoomed_images")
packageAndUpload(rotation_images, "rotation_images")
packageAndUpload(widthShift_images, "widthShift_images")
packageAndUpload(heightShift_images, "heightShift_images")
packageAndUpload(shear_images, "shear_images")
packageAndUpload(horizontalFlip_images, "horizontalFlip_images")
packageAndUpload(verticalFlip_images, "verticalFlip_images")
packageAndUpload(fullyAugmented_images, "fullyAugmented_images")













print("done")



