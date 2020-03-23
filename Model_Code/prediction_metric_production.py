from collections import defaultdict
import numpy as np
import pandas as pd
import os
from blob_utils import *


def thresholdValue(value, threshold):
    if value >= threshold:
        return 1
    else:
        return 0

def divisionCatcher(nuerator, denominator):
    if denominator == 0:
        return 0
    return nuerator/denominator
def thresholdModelStats(results, threshold):
    results['rounded_value'] = results['pred'].apply(thresholdValue, threshold=threshold)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, item in results.iterrows():
        if item['true'].astype('int') == 0 :
            if item['rounded_value'].astype('int') == 0:
                TN +=1
            if item['rounded_value'].astype('int') == 1:
                FP +=1
        if item['true'].astype('int') == 1:
            if item['rounded_value'].astype('int') == 0:
                FN +=1
            if item['rounded_value'].astype('int') == 1:
                TP +=1
    # Calculate metrics NOTE: Add extra calulcations here. Be sure to add a space in the return function, and include a placeholder for the "Best Metric" stat if you are including it.
    P = divisionCatcher(TP,(TP+FP))
    R = TP/(TP+FN)
    I = R + (TN/(TN+FP)) - 1
    FNR = divisionCatcher(FN,(TP+FN))
    return([threshold, TP, TN, FP,FN, P, R, I, FNR,0,0,0,0])

path = "/Users/max/Documents/MMAI/MMAI 894 - Deep Learning/Final Project/prediction results"
experiments = [1,2]


threholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for experiment in experiments:
    filePath = "{}/Experiment {}".format(path,experiment)
    results = []
    for file in os.listdir(filePath):
        if file.endswith(".csv"):
            data = pd.read_csv(os.path.join(filePath, file), index_col=0)
            prelim_results = []
            for threhold in threholds:
                result = thresholdModelStats(data, threhold)
                result.insert(0,file)
                prelim_results.append(result)
            # Gather metrics from each threshold
            prec = [i[6] for i in prelim_results]
            rec = [i[7] for i in prelim_results]
            inf = [i[8] for i in prelim_results]
            fnrs = [i[9] for i in prelim_results]
            # Determine best metric
            best_prec = max(prec)
            best_rec = max(rec)
            best_inf = max(inf)
            best_fnr = min(fnrs)
            # Mark which thresholds are the best based on each metric
            for i in range(len(prelim_results)):
                row = prelim_results[i]
                if row[6] >= best_prec:
                    prelim_results[i][10] = 1
                if row[7] >= best_rec:
                    prelim_results[i][11] = 1
                if row[8] >= best_inf:
                    prelim_results[i][12] =1
                if row[9] <= best_fnr:
                    prelim_results[i][13]=1
                # prelim_results[best_prec][9] == 1
                # prelim_results[best_rec][10] == 1
                # prelim_results[best_inf][11] ==1
            results.extend(prelim_results)
    # NOTE: Make sure to include any new metrics or best values in the columns list
    results_df = pd.DataFrame(results,columns = ["model", "threshold", "TP", "TN", "FP", "FN", "Precision", "Recall", "Informedness", "False Negative Rate","Best Precision", "Best Recall", "Best Informedness", "Best False Negative Rate"])
    results_df.to_csv("/Users/max/Documents/MMAI/MMAI 894 - Deep Learning/Final Project/prediction results/experiment_{}_results.csv".format(str(experiment)))

print("stop")