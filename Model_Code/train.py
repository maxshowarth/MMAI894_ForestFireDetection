import numpy as np
from keras.callbacks import EarlyStopping

import model_plot_utils as mpu
from blob_utils import *
from image_load_split_augment import load_augmented_dataset
from model_build import build_vgg_trainable, build_mobilenetv2, build_nasnet, build_resnetv2, build_efficientnet, \
    build_watts

BUCKET_NAME = "citric-inkwell-268501"


def train_vgg16():
    """
    Build, train and save VGG16 model with block4 and block5 fine-tuning.
    :return: null
    """

    # load data
    training_sets = load_augmented_dataset()

    # build models
    model_vgg = build_vgg_trainable()

    baseWeights_t = model_vgg.get_weights()

    # NOTE: You can still leave this alone if you've only downloaded the fully augmented set.
    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))
        model_vgg.set_weights(baseWeights_t)  # Resets model
        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=2)
        history = model_vgg.fit(train_x, train_y, batch_size=32, epochs=20, verbose=1, validation_split=0.2,
                                shuffle=True,
                                callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history,
                               "./model_cache/train_data/{}_block4and5_vgg16_plots.png".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_block4and5_vgg16_plots.png".format(str(training_set)),
                    "model_charts/{}_block4and5_vgg16_plots.png".format(str(training_set)))

        model_vgg.save("./model_cache/train_data/{}_block4and5_vgg16.h5".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_block4and5_vgg16.h5".format(str(training_set)),
                    "saved_models/{}_block4and5_vgg16.h5".format(str(training_set)))


def train_mobilenetv2():
    """
    Build, train and save MobileNetV2 model with 80% fine-tuning.
    :return: null
    """

    # load data
    training_sets = load_augmented_dataset()

    # build models
    model_mobile = build_mobilenetv2()

    # store base weights
    baseWeights_t = model_mobile.get_weights()

    # NOTE: You can still leave this alone if you've only downloaded the fully augmented set.
    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))
        model_mobile.set_weights(baseWeights_t)  # Resets model
        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=2)
        history = model_mobile.fit(train_x, train_y, batch_size=32, epochs=20, verbose=1, validation_split=0.2,
                                   shuffle=True,
                                   callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history,
                               "./model_cache/train_data/{}_mobilenetv2_plots.png".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_mobilenetv2_plots.png".format(str(training_set)),
                    "model_charts/{}_mobilenetv2_plots.png".format(str(training_set)))

        model_mobile.save("./model_cache/train_data/{}_mobilenetv2.h5".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_mobilenetv2.h5".format(str(training_set)),
                    "saved_models/{}_mobilenetv2.h5".format(str(training_set)))


def train_nasnetmobile():
    """
    Build, train and save NASNetMobile model with no fine-tuning.
    :return: null
    """

    # load data
    training_sets = load_augmented_dataset()

    # build models
    model_nas = build_nasnet()

    baseWeights_t = model_nas.get_weights()

    # NOTE: You can still leave this alone if you've only downloaded the fully augmented set.
    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))
        model_nas.set_weights(baseWeights_t)  # Resets model
        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=2)
        history = model_nas.fit(train_x, train_y, batch_size=32, epochs=20, verbose=1, validation_split=0.2,
                                shuffle=True,
                                callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history,
                               "./model_cache/train_data/{}_nasnetmobile_plots.png".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_nasnetmobile_plots.png".format(str(training_set)),
                    "model_charts/{}_nasnetmobile_plots.png".format(str(training_set)))

        model_nas.save("./model_cache/train_data/{}_nasnetmobile.h5".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_nasnetmobile.h5".format(str(training_set)),
                    "saved_models/{}_nasnetmobile.h5".format(str(training_set)))


def train_resnet():
    """
    Build, train and save Resnet model.
    :return: null
    """

    # load data
    training_sets = load_augmented_dataset()

    # build models
    model_resnet = build_resnetv2()

    baseWeights_t = model_resnet.get_weights()

    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))
        model_resnet.set_weights(baseWeights_t)  # Resets model
        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=3)
        history = model_resnet.fit(train_x, train_y, batch_size=32, epochs=40, verbose=1, validation_split=0.2,
                                   shuffle=True,
                                   callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history,
                               "./model_cache/train_data/{}_resnet_plots.png".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_resnet_plots.png".format(str(training_set)),
                    "model_charts/{}_resnet_plots.png".format(str(training_set)))

        model_resnet.save("./model_cache/train_data/{}_resnet.h5".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_resnet.h5".format(str(training_set)),
                    "saved_models/{}_resnet.h5".format(str(training_set)))


def train_efficientnet():
    """
    Build, train and save Efficientnet model.
    :return: null
    """

    # load data
    training_sets = load_augmented_dataset()

    # build models
    model_efn = build_efficientnet()

    baseWeights_t = model_efn.get_weights()

    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))
        model_efn.set_weights(baseWeights_t)  # Resets model
        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=5)
        history = model_efn.fit(train_x, train_y, batch_size=32, epochs=40, verbose=1, validation_split=0.2,
                                shuffle=True,
                                callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history,
                               "./model_cache/train_data/{}_efficientnet_plots.png".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_efficientnet_plots.png".format(str(training_set)),
                    "model_charts/{}_efficientnet_plots.png".format(str(training_set)))

        model_efn.save("./model_cache/train_data/{}_efficientnet.h5".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_efficientnet.h5".format(str(training_set)),
                    "saved_models/{}_efficientnet.h5".format(str(training_set)))


def train_watts():
    """
    Build, train and save simple CNN model from scatch.
    :return: null
    """

    # load data
    training_sets = load_augmented_dataset()

    # build models
    model = build_watts()

    for training_set in training_sets:
        print("     Starting training for set {}".format(str(training_set)))

        train_x = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][0]))
        train_y = np.load(os.path.join("./model_cache/train_data", training_sets[training_set][1]))

        early_stopping_monitor = EarlyStopping(patience=5)
        history = model.fit(train_x, train_y, batch_size=32, epochs=40, verbose=1, validation_split=0.2,
                            shuffle=True,
                            callbacks=[early_stopping_monitor])

        mpu.plot_accuracy_loss(history,
                               "./model_cache/train_data/{}_watts_plots.png".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_watts_plots.png".format(str(training_set)),
                    "model_charts/{}_watts_plots.png".format(str(training_set)))

        model.save("./model_cache/train_data/{}_watts.h5".format(str(training_set)))

        upload_blob(BUCKET_NAME, "./model_cache/train_data/{}_watts.h5".format(str(training_set)),
                    "saved_models/{}_watts.h5".format(str(training_set)))


if __name__ == '__main__':
    train_vgg16()
    train_mobilenetv2()
    train_nasnetmobile()
    train_resnet()
    train_efficientnet()
    train_watts()
