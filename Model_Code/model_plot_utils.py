import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def plot_accuracy_loss(history, path):
    """
    Plot the accuracy and the loss during the training the network
    :param history: model training history object
    :param path: path to save figure
    :return: null
    """
    fig = plt.figure(figsize=(15, 10))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label="acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label="val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'], 'bo--', label="loss")
    plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    fig.savefig(path)

    plt.legend()
    plt.close(fig)


def showNumpyImage(image):
    """
    Displays a numpy image object in true color.

    :param image:
    :return: null
    """

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.show()


def display_examples(class_names, images, labels):
    """
    Display 25 images from the images array with its corresponding labels

    :param class_names: list-like object containing class names
    :param images: list-like object containing images database
    :param labels: list-like object containing labels
    :return: null
    """

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):
    """
    Print 25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels.

    :param class_names: list-like object containing class names
    :param test_images: list-like object containing images
    :param test_labels: list-like object containing true labels of images
    :param pred_labels: list-like object containing predicted labels of images
    :return: null
    """

    matchlabeled = (test_labels == pred_labels)
    mislabeled_indices = np.where(matchlabeled == 0)
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]

    title = "Some examples of mislabeled images by the classifier:"
    display_examples(class_names, mislabeled_images, mislabeled_labels)
