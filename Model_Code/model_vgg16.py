import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.applications import vgg16
import matplotlib.pyplot as plt
import time
from PIL import Image
import pandas as pd
import model_evaluation_utils as meu
from google.cloud import storage
import os
from collections import defaultdict

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GCP Playground-34c3d1faef3b.json"
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
if os.path.isdir('./model_cache/VGG16_cache'):
    print("Model Cache Exists")
else:
    os.makedirs("./model_cache/VGG16_cache")
    print("Created Model Cache")

bucket_files = [blob.name for blob in storage_client.list_blobs(bucket_name)]

# Get available training sets from cloud and download
training_sets = defaultdict(list)
for set in bucket_files:
    if "training_sets" in set:
        training_sets[set.split("/")[1]].append(set.replace("/","-"))
        if os.path.exists(os.path.join("./model_cache/VGG16_cache", str(set.replace("/","-")))):
            print("{} already downloaded".format(str(set.split("/")[1])))
        else:
            print("{}  downloading".format(str(set.split("/")[1])))
            # download_blob(bucket_name, set, os.path.join("./model_cache/VGG16_cache", str(set.replace("/","-"))))
    else:
        continue


# Base VGG16 Model No Retraining, Imagenet Weights
print("")
print("Beginning base VGG16 Training")
# input_shape = (224, 224, 3)
# model_vgg16 = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)
# output = model_vgg16.layers[-1].output
# output = keras.layers.Flatten()(output)
# vgg_model = Model(model_vgg16.input, output)
#
# model = Sequential()
# model.add(vgg_model)
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-5),
#               metrics=['accuracy'])
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-5),
#               metrics=['accuracy'])

# Save base model weights
# baseWeights = model.get_weights()
# Loop and train using each training set
for training_set in training_sets:
    print("     Starting training for set {}".format(str(training_set)))
    # model.set_weights(baseWeights)
    train_x = np.load(os.path.join("./model_cache/VGG16_cache", training_sets[training_set][0]))
    train_y = np.load(os.path.join("./model_cache/VGG16_cache", training_sets[training_set][1]))
    # history = model.fit(train_x_scaled, train_y, batch_size=32, epochs=10, verbose=1)
    # model.save("./model_cache/VGG16_cache/{}_base_vgg16.h5".format(str(training_set)))
    # upload_blob(bucket_name,"./model_cache/VGG16_cache/{}_base_vgg16.h5".format(str(training_set)),"{}_base_vgg16.h5".format(str(training_set)))


# Vase VGG16 Model Retrain Block 4 and 5, Imagenet Weights
print("")
print("Beginning retrainable VGG16 Training")

input_shape = (224, 224, 3)
model_vgg16 = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)
output = model_vgg16.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(model_vgg16.input, output)

# Set blocks 4 and 5 to be fine tuneable
vgg_model.trainable = True
set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

for training_set in training_sets:
    print("     Starting training for set {}".format(str(training_set)))
    model.set_weights(baseWeights)
    train_x = np.load(os.path.join("./model_cache/VGG16_cache", training_set[0]))
    train_y = np.load(os.path.join("./model_cache/VGG16_cache", training_set[1]))
    history = model.fit(train_x_scaled, train_y, batch_size=32, epochs=10, verbose=1)
    model.save("./model_cache/VGG16_cache/{}_base_vgg16.h5".format(str(training_set)))
    upload_blob(bucket_name,"./model_cache/VGG16_cache/{}_block4and5_vgg16.h5".format(str(training_set)),"{}_base_vgg16.h5".format(str(training_set)))




print(varname(variable))
