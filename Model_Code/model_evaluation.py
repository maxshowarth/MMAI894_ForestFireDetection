from collections import defaultdict

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import efficientnet.tfkeras
#
from blob_utils import *

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GCP Playground-34c3d1faef3b.json"
BUCKET_NAME = "citric-inkwell-268501"

# Open gs Client
storage_client = storage.Client()
bucket_files = [blob.name for blob in storage_client.list_blobs(BUCKET_NAME)]


def load_test_dataset():
    """
    Loads testing dataset from cloud storage.

    :return: test_x, test_y
    """

    # Check for existance of local model_cache and create if it does not exist
    if os.path.isdir('./model_cache/test_data'):
        print("Test Data Exists")
    else:
        os.makedirs("./model_cache/test_data")
        print("Created Test Data")

    test_sets = defaultdict(list)
    for filename in bucket_files:
        if "test_set" in filename:
            test_sets[filename.split("/")[1]].append(filename.replace("/", "-"))
            if os.path.exists(os.path.join("./model_cache/test_data", str(filename.replace("/", "-")))):
                print("{} already downloaded".format(str(filename.split("/")[1])))
            else:
                print("{}  downloading".format(str(filename.split("/")[1])))
                download_blob(BUCKET_NAME, filename, os.path.join("./model_cache/test_data", str(filename.replace("/", "-"))))
        else:
            continue

    # load testset
    test_x = np.load(os.path.join("./model_cache/test_data", test_sets['test_x.npy'][0]))
    test_y = np.load(os.path.join("./model_cache/test_data", test_sets['test_y.npy'][0]))
    return test_x, test_y


def load_saved_models():
    """
    Loads saved models (hd5) from cloud storage.

    :return: saved models
    """

    # load saved models:
    if os.path.isdir('./model_cache/saved_models'):
        print("Saved models Exists")
    else:
        os.makedirs("./model_cache/saved_models")
        print("Created saved models")

    saved_models = defaultdict(list)
    for filename in bucket_files:
        if "saved_models" in filename:
            saved_models[filename.split("/")[1]].append(filename.replace("/", "-"))
            if os.path.exists(os.path.join("./model_cache/saved_models", str(filename.replace("/", "-")))):
                print("{} already downloaded".format(str(filename.split("/")[1])))
            else:
                print("{}  downloading".format(str(filename.split("/")[1])))
                download_blob(BUCKET_NAME, filename, os.path.join("./model_cache/saved_models",
                                                                  str(filename.replace("/", "-"))))
        else:
            continue

    return saved_models


def prediction(models_to_evaluate=None):
    """
    Makes predictions and uploads results to cloud storage

    :return: null
    """
    test_x, test_y = load_test_dataset()
    saved_models = load_saved_models()

    if models_to_evaluate is None:
        models_to_evaluate = [i for i in saved_models.values()]
    else:
        models_to_evaluate = [i for i in models_to_evaluate]

    if os.path.isdir('./model_cache/preds'):
        print("folder Exists")
    else:
        os.makedirs("./model_cache/preds")


    # Evaluate all models on the testset and upload the predictions
    for saved_model in models_to_evaluate:
        model = load_model(os.path.join("./model_cache/saved_models", saved_model))

        test_predictions = model.predict(test_x)

        preds = pd.DataFrame({'true': test_y, 'pred': np.ravel(test_predictions)})

        preds.to_csv("./model_cache/preds/{}_pred.csv".format(str(saved_model)[13:-3]))
        upload_blob(BUCKET_NAME, "./model_cache/preds/{}_pred.csv".format(str(saved_model[13:-3])), "models_predictions/{}_pred.csv".format(str(saved_model[13:-3])))


if __name__ == '__main__':
    prediction()
