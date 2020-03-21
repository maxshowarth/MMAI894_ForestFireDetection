from keras.models import load_model

import os
import model_evaluation_utils as meu
import model_plot_utils as mpu
from collections import defaultdict

from blob_utils import *



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GCP Playground-34c3d1faef3b.json"
storage_client = storage.Client()

bucket_name = "citric-inkwell-268501"
bucket_files = [blob.name for blob in storage_client.list_blobs(bucket_name)]


def load_testset():
    # Check for existance of local model_cache and create if it does not exist
    if os.path.isdir('./model_cache/test_data'):
        print("Test Data Exists")
    else:
        os.makedirs("./model_cache/test_data")
        print("Created Test Data")


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

            
    #laod testset        
    test_x = np.load(os.path.join("./model_cache/test_data", test_sets['test_x.npy'][0]))
    test_y = np.load(os.path.join("./model_cache/test_data", test_sets['test_y.npy'][0]))
    return test_x, test_y

def load_saved_models():
    #load saved models:
    if os.path.isdir('./model_cache/saved_models'):
        print("Saved models Exists")
    else:
        os.makedirs("./model_cache/saved_models")
        print("Created saved models")


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

    return saved_models   

def prediction():
    
    if os.path.isdir('./model_cache/preds'):
        print("folder Exists")
    else:
        os.makedirs("./model_cache/preds")
              
    
    test_x, test_y = load_testset()
    saved_models = load_saved_models()
    
    # Evaluate all models on the testset and upload the predictions
    for saved_model in saved_models.values():
        model = load_model(os.path.join("./model_cache/saved_models", saved_model[0]))
        
        test_predictions = model.predict(test_x)
        
        preds = pd.DataFrame({'true':test_y, 'pred':np.ravel(test_predictions)})
        
        preds.to_csv("./model_cache/preds/{}_pred.csv".format(str(saved_model[0])[13:-3]))                   
        upload_blob(bucket_name,"./model_cache/preds/{}_pred.csv".format(str(saved_model[0])[13:-3]),"models_predictions/{}_pred.csv".format(str(saved_model[0])[13:-3]))

    
if __name__ == '__main__':   
    prediction()        
 