import keras
import tensorflow as tf
import time
from PIL import Image
from keras import optimizers
from keras.applications import mobilenet_v2, nasnet, vgg16
from keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Flatten
from keras.models import Model, Sequential


def set_trainable(model, fine_tune):
    """
    Configure model layers to be trainable and freeze remainder of weights.
    :param model: Model to be configured
    :param fine_tune: Layers to be trained
    :return: Configured model
    """

    # Set blocks 15 and 16 to be fine-tuneable
    model.trainable = True
    trainable_flag = False
    for layer in model.layers:
        if layer.name in fine_tune:
            trainable_flag = True
        if trainable_flag:
            layer.trainable = True
        else:
            layer.trainable = False

    # layers = [(layer, layer.name, layer.trainable) for layer in mobile_model.layers]
    # pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
    # ps.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']).tail(25)

    return model


def add_head(transfer_model, input_shape, head_type):
    """
    Adds a classification head to pre-trained model architecture
    :param head_type: Classifier head type
    :param transfer_model: model to add head to
    :param input_shape: Shape of input layer
    :return: model with classification head
    """
    if head_type == "basic":
        output = transfer_model.layers[-1].output
        output = keras.layers.Flatten()(output)
        joined_model = Model(transfer_model.input, output)

        model = Sequential()
        model.add(joined_model)
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        return model

    elif head_type == "colin":
        inputs = Input(input_shape)
        x = transfer_model(inputs)
        out1 = GlobalMaxPooling2D()(x)
        out2 = GlobalAveragePooling2D()(x)
        out3 = Flatten()(x)
        out = Concatenate(axis=-1)([out1, out2, out3])
        out = Dropout(0.5)(out)
        out = Dense(1, activation="sigmoid", name="3_")(out)
        model = Model(inputs, out)
        return model

    else:
        print("Invalid model head type.")
        exit()


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

    # Configure model layers to be trainable
    mobile_model = set_trainable(model_mobilenet, fine_tune)

    # add custom classifier head to model
    model = add_head(mobile_model, input_shape, head_type='colin')

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

    # Configure model layers to be trainable
    nas_model = set_trainable(model_nasnet, fine_tune)

    # Add classification head
    model = add_head(nas_model, input_shape, head_type='colin')

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
    vgg_model_t = Model(base_model.input, output)

    # configure model layers to be trainable
    model_t = set_trainable(vgg_model_t, fine_tune)

    # add classification head to model
    model_t = add_head(model_t, input_shape, head_type='basic')

    # compile model
    model_t.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-5),
                    metrics=['accuracy'])

    print(model_t.summary())

    return model_t


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
    vgg_model_t = Model(base_model.input, output)

    # configure model layers to be trainable
    model_t = set_trainable(vgg_model_t, fine_tune)

    # add classification head to model
    model_t = add_head(model_t, input_shape, head_type='basic')

    # compile model
    model_t.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-5),
                    metrics=['accuracy'])

    print(model_t.summary())

    return model_t
