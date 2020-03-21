import cv2
import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from PIL import Image
from keras import optimizers
from keras.applications import mobilenet_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential, load_model


def train_model(model, X, y, batch_size=32, epochs=5):
    """
    Trains a model
    :param model: Compiled model to be trained
    :param X: Training features, scaled
    :param y: Training labels
    :param batch_size: Batch size for training
    :param epochs: Training epochs
    :return: Trained model, Training history
    """
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs,
                        verbose=True)

    return model, history


def evaluate_model(model, X, y):
    """
    Evaluates a model
    :param model: Trained model to be evaluated
    :param X: Testing features, scaled
    :param y: Testing labels
    :return:
    """
    # Make predictions with model and convert predictions into binary
    test_predictions = model.predict(X)
    test_predictions_labelled = [0 if x < 0.1 else 1 for x in test_predictions]

    # Display performance metrics
    meu.display_model_performance_metrics(true_labels=y, predicted_labels=test_predictions_labelled,
                                          classes=list(set(y)))

    return model
