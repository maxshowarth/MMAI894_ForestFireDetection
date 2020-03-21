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
from google.cloud import storage
import os
from collections import defaultdict
from keras.models import load_model


import model_evaluation_utils as meu
import model_plot_utils as mpu
from blob_utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GCP Playground-34c3d1faef3b.json"
storage_client = storage.Client()

bucket_name = "citric-inkwell-268501"



# Check for existance of local model_cache and create if it does not exist
if os.path.isdir('./model_cache/train_data'):
    print("Model Cache Exists")
else:
    os.makedirs("./model_cache/train_data")
    print("Created Model Cache")

bucket_files = [blob.name for blob in storage_client.list_blobs(bucket_name)]

# Get available training sets from cloud and download
### NOTE: All training sets are being downloaded and used. If you only want to train on the fully augmented set uncomment the line below
bucket_files = ['training_sets/full_augmentation/full_augmentation_train_x_aug.npy', 'training_sets/full_augmentation/full_augmentation_train_y_aug.npy',
               'training_sets/no_augmentation/no_augmentation_train_x.npy',
                'training_sets/no_augmentation/no_augmentation_train_y.npy']

training_sets = defaultdict(list)
for set in bucket_files:
    if "training_sets" in set:
        training_sets[set.split("/")[1]].append(set.replace("/","-"))
        if os.path.exists(os.path.join("./model_cache/train_data", str(set.replace("/","-")))):
            print("{} already downloaded".format(str(set.split("/")[1])))
        else:
            print("{}  downloading".format(str(set.split("/")[1])))
            download_blob(bucket_name, set, os.path.join("./model_cache/train_data", str(set.replace("/","-"))))
    else:
        continue


###################################
########### Train Models###########

def vgg_model_train():


    # Base VGG16 Model No Retraining, Imagenet Weights
    ### NOTE: This is where you should insert your model code
    print("")
    print("Beginning base VGG16 Training")
    input_shape = (224, 224, 3)
    model_vgg16 = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)
    output = model_vgg16.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model = Model(model_vgg16.input, output)

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

    # Save base model weights so the model can be reset after each training
    baseWeights = model.get_weights()
    # Loop and train using each training set
    ### NOTE: You can still leave this alone if you've only downloaded the fully augmented set.
    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))
        model.set_weights(baseWeights) # Resets model
        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=2)
        history = model.fit(train_x, train_y, batch_size=32, epochs=20, verbose=1, validation_split = 0.2, shuffle=True, callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history, "./model_cache/train_data/{}_base_vgg16_plots.png".format(str(training_set)))
        upload_blob(bucket_name,"./model_cache/train_data/{}_base_vgg16_plots.png".format(str(training_set)),"model_charts/{}_base_vgg16_plots.png".format(str(training_set)))

        model.save("./model_cache/train_data/{}_base_vgg16.h5".format(str(training_set)))
        upload_blob(bucket_name,"./model_cache/train_data/{}_base_vgg16.h5".format(str(training_set)),"saved_models/{}_base_vgg16.h5".format(str(training_set)))


    # Vase VGG16 Model Retrain Block 4 and 5, Imagenet Weights
    ### NOTE: This is where you should insert your model code

    print("")
    print("Beginning retrainable VGG16 Training")

    input_shape = (224, 224, 3)
    model_vgg16_t = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)
    output = model_vgg16_t.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model_t = Model(model_vgg16_t.input, output)

    # Set blocks 4 and 5 to be fine tuneable
    vgg_model_t.trainable = True
    set_trainable = False
    for layer in vgg_model_t.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model_t = Sequential()
    model_t.add(vgg_model_t)
    model_t.add(Dense(512, activation='relu', input_dim=input_shape))
    model_t.add(Dropout(0.3))
    model_t.add(Dense(512, activation='relu'))
    model_t.add(Dropout(0.3))
    model_t.add(Dense(1, activation='sigmoid'))

    model_t.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    baseWeights_t = model_t.get_weights()


    ### NOTE: You can still leave this alone if you've only downloaded the fully augmented set.
    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))
        model_t.set_weights(baseWeights_t) # Resets model
        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=2)
        history = model_t.fit(train_x, train_y, batch_size=32, epochs=20, verbose=1, validation_split = 0.2, shuffle=True, callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history, "./model_cache/train_data/{}_block4and5_vgg16_plots.png".format(str(training_set)))

        upload_blob(bucket_name,"./model_cache/train_data/{}_block4and5_vgg16_plots.png".format(str(training_set)),"model_charts/{}_block4and5_vgg16_plots.png".format(str(training_set)))

        model_t.save("./model_cache/train_data/{}_block4and5_vgg16.h5".format(str(training_set)))

        upload_blob(bucket_name,"./model_cache/train_data/{}_block4and5_vgg16.h5".format(str(training_set)),"saved_models/{}_block4and5_vgg16.h5".format(str(training_set)))


if __name__ == '__main__':
    vgg_model_train()

