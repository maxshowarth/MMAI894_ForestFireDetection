from keras.models import load_model

import os
import model_evaluation_utils as meu
import model_plot_utils as mpu
from collections import defaultdict

from blob_utils import *



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GCP Playground-34c3d1faef3b.json"
storage_client = storage.Client()

bucket_name = "citric-inkwell-268501"



# Check for existance of local model_cache and create if it does not exist
if os.path.isdir('./model_cache/test_data'):
    print("Test Data Exists")
else:
    os.makedirs("./model_cache/test_data")
    print("Created Test Data")

bucket_files = [blob.name for blob in storage_client.list_blobs(bucket_name)]
#bucket_files = ['test_set/test_x.npy', 'test_set/test_y.npy']


test_sets = defaultdict(list)
for set in bucket_files:
    if "test_set" in set:
        test_sets[set.split("/")[1]].append(set.replace("/","-"))
        if os.path.exists(os.path.join("./model_cache/test_data", str(set.replace("/","-")))):
            print("{} already downloaded".format(str(set.split("/")[1])))
        else:
            print("{}  downloading".format(str(set.split("/")[1])))
            download_blob(bucket_name, set, os.path.join("./model_cache/test_data", str(set.replace("/","-"))))
    else:
        continue
        

        
#load saved models:

if os.path.isdir('./model_cache/saved_models'):
    print("Saved models Exists")
else:
    os.makedirs("./model_cache/saved_models")
    print("Created saved models")

#bucket_files = [blob.name for blob in storage_client.list_blobs(bucket_name)]

saved_models = defaultdict(list)
for set in bucket_files:
    if "saved_models" in set:
        saved_models[set.split("/")[1]].append(set.replace("/","-"))
        if os.path.exists(os.path.join("./model_cache/saved_models", str(set.replace("/","-")))):
            print("{} already downloaded".format(str(set.split("/")[1])))
        else:
            print("{}  downloading".format(str(set.split("/")[1])))
            download_blob(bucket_name, set, os.path.join("./model_cache/saved_models", str(set.replace("/","-"))))
    else:
        continue

if os.path.isdir('./model_cache/reports'):
    print("folder Exists")
else:
    os.makedirs("./model_cache/reports")

        
thershold = 0.3        

test_x = np.load(os.path.join("./model_cache/test_data", test_sets['test_x.npy'][0]))
test_y = np.load(os.path.join("./model_cache/test_data", test_sets['test_y.npy'][0]))
    
for saved_model in saved_models.values():
    model = load_model(os.path.join("./model_cache/saved_models", saved_model[0]))
    test_predictions = model.predict(test_x)
    test_predictions_labelled = [0 if x<thershold else 1 for x in test_predictions]


    report = meu.classification_report_df(true_labels=test_y, predicted_labels=test_predictions_labelled, classes=[1,0])
    report.to_csv("./model_cache/reports/{}_report.csv".format(str(saved_model[0])[:-3]))                   
    upload_blob(bucket_name,"./model_cache/reports/{}_report.csv".format(str(saved_model[0])[:-3]),"model_reports/{}_report.csv".format(str(saved_model[0])[:-3]))

