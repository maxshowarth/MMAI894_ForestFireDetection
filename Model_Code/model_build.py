import keras
import tensorflow as tf
import time
from PIL import Image
from keras import optimizers
from keras.applications import mobilenet_v2, nasnet
from keras.layers import Dropout, Dense
from keras.models import Model, Sequential


def set_trainable(model, fine_tune):
    """
    Configure model layers to be trainable and freeze remainder of weights.
    :param model: Model to be configured
    :param fine_tune: Layers to be trained
    :return: Configured model
    """
    output = model.layers[-1].output
    output = keras.layers.Flatten()(output)
    joined_model = Model(model.input, output)

    # Set blocks 15 and 16 to be fine-tuneable
    joined_model.trainable = True
    set_trainable = False
    for layer in joined_model.layers:
        if layer.name in fine_tune:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # layers = [(layer, layer.name, layer.trainable) for layer in mobile_model.layers]
    # pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
    # ps.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']).tail(25)

    return joined_model


def add_head(transfer_model, input_shape):
    """
    Adds a classification head to pre-trained model architecture
    :param transfer_model: model to add head to
    :param input_shape: Shape of input layer
    :return: model with classification head
    """
    # add custom classifier head to model
    model = Sequential()
    model.add(transfer_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_mobilenetv2(weights='imagenet', fine_tune=None):
    """
    Builds and compiles the MobileNetV2 model using pre-trained weights and custom classification head.

    :param weights: Pre-trained weights to be used
    :param fine_tune: First layer of each block to be fine-tuned
    :return: The compiled model
    """
    if fine_tune is None:
        fine_tune = ['block_14_expand', 'block_15_expand', 'block_16_expand']
    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    model_mobilenet = mobilenet_v2.MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)

    # Configure model layers to be trainable
    mobile_model = set_trainable(model_mobilenet, fine_tune)

    # add custom classifier head to model
    model = add_head(mobile_model, input_shape)

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
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
        fine_tune = ['block_14_expand', 'block_15_expand', 'block_16_expand']
    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    model_nasnet = nasnet.NASNetMobile(include_top=False, weights=weights, input_shape=input_shape)

    # Configure model layers to be trainable
    nas_model = set_trainable(model_nasnet, fine_tune)

    # Add classification head
    model = add_head(nas_model, input_shape)

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model
