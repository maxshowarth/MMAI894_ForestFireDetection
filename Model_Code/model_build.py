import keras
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from PIL import Image
from keras import optimizers
from keras.applications import mobilenet_v2, nasnet
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential, load_model


def build_mobilenetv2(weights='imagenet', fine_tune=['block_14_expand', 'block_15_expand', 'block_16_expand']):
    """
    Builds and compiles the MobileNetV2 model using pre-trained weights and custom classification head.

    :param weights: Pre-trained weights to be used
    :param fine_tune: First layer of each block to be fine-tuned
    :return: The compiled model
    """
    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    model_mobilenet = mobilenet_v2.MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
    output = model_mobilenet.layers[-1].output
    output = keras.layers.Flatten()(output)
    mobile_model = Model(model_mobilenet.input, output)

    # Set blocks 15 and 16 to be fine-tuneable
    mobile_model.trainable = True
    set_trainable = False
    for layer in mobile_model.layers:
        if layer.name in fine_tune:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # layers = [(layer, layer.name, layer.trainable) for layer in mobile_model.layers]
    # pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
    # ps.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']).tail(25)

    # add custom classifier head to model
    model = Sequential()
    model.add(mobile_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model


def build_nasnet(weights='imagenet', fine_tune=['block_14_expand', 'block_15_expand', 'block_16_expand']):
    """
    Builds and compiles the NASNetV2 model using pre-trained weights and custom classification head.

    :param weights: Pre-trained weights to be used
    :param fine_tune: First layer of each block to be fine-tuned
    :return: The compiled model
    """
    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    model_nasnet = nasnet.NASNetMobile(include_top=False, weights=weights, input_shape=input_shape)
    output = model_nasnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    nas_model = Model(model_nasnet.input, output)

    # Set blocks 15 and 16 to be fine-tuneable
    nas_model.trainable = True
    set_trainable = False
    for layer in nas_model.layers:
        if layer.name in fine_tune:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # layers = [(layer, layer.name, layer.trainable) for layer in mobile_model.layers]
    # pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
    # ps.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']).tail(25)

    # add custom classifier head to model
    model = Sequential()
    model.add(nas_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model
