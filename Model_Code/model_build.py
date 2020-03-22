# !pip install efficientnet
import efficientnet.keras as efn
import keras
from keras import optimizers
from keras.applications import mobilenet_v2, nasnet, vgg16, resnet_v2
from keras.layers import Conv2D, MaxPooling2D
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
    Build VGG16 model with fine-tuning.

    :param weights: Pre-trained weights to be used.
    :param fine_tune: Layers to be fine-tuned.
    :return: The compiled model
    """

    if fine_tune is None:
        fine_tune = ['block4_conv1', 'block5_conv1']

    input_shape = (224, 224, 3)

    # configure base model
    base_model = vgg16.VGG16(include_top=False, weights=weights, input_shape=input_shape)
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model = Model(base_model.input, output)

    # Set blocks 4 and down to be fine tuneable
    vgg_model.trainable = True
    set_trainable = False
    for layer in vgg_model.layers:
        if layer.name in fine_tune:
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

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    print(model.summary())

    return model


def build_vgg_frozen(weights='imagenet', fine_tune=None):
    """
    Build VGG16 with no fine-tuning.

    :param weights: Pre-trained weights to be used.
    :param fine_tune: Layers to be fine-tuned.
    :return: The compiled model
    """

    if fine_tune is None:
        fine_tune = ['']

    input_shape = (224, 224, 3)

    # configure base model
    base_model = vgg16.VGG16(include_top=False, weights=weights, input_shape=input_shape)
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model = Model(base_model.input, output)

    # Freeze pre-trained weights
    vgg_model.trainable = False

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


def build_resnetv2(weights='imagenet', fine_tune=None):
    """
    Builds and compiles the ResNet152V2 model using pre-trained weights and custom classification head.

    :param weights: Pre-trained weights to be used
    :param fine_tune: layers to be fine-tuned
    :return: The compiled model
    """

    if fine_tune is None:
        fine_tune = ['conv5_block3_3_conv', 'post_relu']
    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    base_model = resnet_v2.ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape)
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    model_resnet = Model(base_model.input, output)

    # Set blocks to be fine tuneable
    model_resnet.trainable = True
    set_trainable = False
    for layer in model_resnet.layers:
        if layer.name in fine_tune:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    model.add(model_resnet)
    # model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model


def build_efficientnet(weights='imagenet', fine_tune=None):
    """
    Builds and compiles the EfficientNetB3 model using pre-trained weights and custom classification head.

    :param weights: Pre-trained weights to be used
    :param fine_tune: layers to be fine-tuned
    :return: The compiled model
    """

    if fine_tune is None:
        fine_tune = 'block6b_add'
    input_shape = (224, 224, 3)

    # Configure base model from pretrained
    base_model = efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    model_efn = Model(base_model.input, output)

    # Set blocks to be fine tuneable
    model_efn.trainable = True
    set_trainable = False
    for layer in model_efn.layers:
        if layer.name == fine_tune:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    model.add(model_efn)
    model.add(Dense(256, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model


def build_watts():
    """
    Builds and compiles the Simple CNN model.

    :return: The compiled model
    """

    input_shape = (224, 224, 3)

    model = Sequential()
    model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=['accuracy'])

    # print model summary
    print(model.summary())

    return model
