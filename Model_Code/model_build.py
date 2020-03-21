import keras
import tensorflow as tf
import time
from PIL import Image
from keras import optimizers
from keras.applications import mobilenet_v2, nasnet, vgg16
from keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Flatten
from keras.models import Model, Sequential


def build_mobilenetv2(weights='imagenet', fine_tune=None):
    """
    Builds and compiles the MobileNetV2 model using pre-trained weights and custom classification head.

    :param weights: Pre-trained weights to be used
    :param fine_tune: First layer of each block to be fine-tuned
    :return: The compiled model
    """
    if fine_tune is None:
        fine_tune = ['block_4_expand']
    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    model_mobilenet = mobilenet_v2.MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)

    # Set blocks 4 and down to be fine tuneable
    model_mobilenet.trainable = True
    set_trainable = False
    for layer in model_mobilenet.layers:
        if layer.name in fine_tune:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    inputs = Input((224, 224, 3))
    x = model_mobilenet(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model


def build_nasnet(weights='imagenet', fine_tune=None):
    """
    Builds and compiles the NASNetV2 model using pre-trained weights and custom classification head.

    :param weights: Pre-trained weights to be used
    :param fine_tune: First layer of each block to be fine-tuned
    :return: The compiled model
    """
    if fine_tune is None:
        fine_tune = ['*']

    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    model_nasnet = nasnet.NASNetMobile(include_top=False, weights=weights, input_shape=input_shape)

    # Set blocks 4 and down to be fine tuneable
    model_nasnet.trainable = True
    set_trainable = False
    for layer in model_nasnet.layers:
        if layer.name in fine_tune:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    inputs = Input((224, 224, 3))
    x = model_nasnet(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model


def build_vgg_trainable(weights='imagenet', fine_tune=None):
    """

    :param weights:
    :param fine_tune:
    :return:
    """
    if fine_tune is None:
        fine_tune = ['block5_conv1', 'block4_conv1']

    input_shape = (224, 224, 3)

    # configure base model
    base_model = vgg16.VGG16(include_top=False, weights=weights, input_shape=input_shape)
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model = Model(base_model.input, output)

    model = Sequential()
    model.add(vgg_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-5),
                    metrics=['accuracy'])

    print(model.summary())

    return model


def build_vgg_frozen(weights='imagenet', fine_tune=None):
    """

    :param weights:
    :param fine_tune:
    :return:
    """
    if fine_tune is None:
        fine_tune = ['']

    input_shape = (224, 224, 3)

    # configure base model
    base_model = vgg16.VGG16(include_top=False, weights=weights, input_shape=input_shape)
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model = Model(base_model.input, output)

    model = Sequential()
    model.add(vgg_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-5),
                    metrics=['accuracy'])

    print(model.summary())

    return model
