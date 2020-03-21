import numpy as np
from keras.callbacks import EarlyStopping

import model_plot_utils as mpu
from blob_utils import *
from image_load_split_augment import load_augmented_dataset
from model_build import build_vgg_trainable, build_mobilenetv2, build_nasnet

BUCKET_NAME = "citric-inkwell-268501"


def train_vgg16():
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
