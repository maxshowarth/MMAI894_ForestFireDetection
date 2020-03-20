#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.applications import vgg16
import matplotlib.pyplot as plt
import time
from PIL import Image
import pandas as pd
import model_evaluation_utils as meu
import model_plot_utils as plot
from keras import optimizers


# Load Data
def load_data(images_npy = "training_data.npy"):
# if you haven't created training_data.npy yet, run load_and_split_data.py first
    data = np.load(images_npy, allow_pickle = True)


    # Split into labels and images then test and train
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 42)


    # Scale images
    train_x_scaled = train_x.astype('float32')
    test_x_scaled = test_x.astype('float32')
    train_x_scaled /= 255
    test_x_scaled /= 255


def load_model(model_name, pre_train_weights = 'imagenet'):
    
    if model_name = 'VGG16':
        
        # Configure base model
        input_shape = (224, 224, 3)
        model_vgg16 = vgg16.VGG16(include_top = False, weights = pre_train_weights, input_shape = input_shape)
        output = model_vgg16.layers[-1].output
        output = keras.layers.Flatten()(output)
        model = Model(model_vgg16.input, output)

        # Set blocks 4 and 5 to be fine tuneable
        model.trainable = True
        set_trainable = False
        for layer in model.layers:
            if layer.name in ['block5_conv1', 'block4_conv1']:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
        pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
        
        
    elif model_name = 'MobileNetV2':
        pass
    
    # add all models here
    
    else:
        raise ValueError('Model_name is not valid. valid models: ["VGG16", "ResNet", "......"]')
    
    return model

def add_last_layers(pretrained_model, input_shape, dropout = 0.3, n_hidden_layers = 2, n_units = 512):   

    model = Sequential()
    model.add(pretrained_model)
    for layer in range(n_hidden_layers):
        if layer = 0:
            model.add(Dense(n_units, activation='relu', input_dim=input_shape))
        else:
            model.add(Dense(n_units, activation='relu'))
        
    model.add(Dense(1, activation='sigmoid'))

    return model 


    
def compile_model(model, loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), metrics=['accuracy']):
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
def fit_model(xtrain, ytrain, batch_size=32, epochs=10, validation_split = 0.2, verbose=1):
    
    history = model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_split = validation_split,
                              verbose=verbose)
    
    return history

