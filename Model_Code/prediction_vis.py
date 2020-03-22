from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report, precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.ylim([0, 1])
    plt.figure(figsize=(8, 8))


def prediction_result(model, threshold=0.5, confusion_mtx=False):
    model['class'] = np.where(model.pred < threshold, 0, 1)
    report = classification_report(model['true'], model['class'], output_dict=True)
    print(classification_report(model['true'], model['class'], output_dict=False))
    df_report = pd.DataFrame(report).transpose()

    if confusion_mtx:
        cm = pd.DataFrame(confusion_matrix(model['true'], model['class']))
        print(cm)
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd')
        tick_marks = np.arange(len(cm))

        plt.show()

    precisions, recalls, thresholds = precision_recall_curve(model['true'], model['pred'])

    plt.figure(figsize=(8, 8))
    sns.set(style="whitegrid")
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.xticks([x / 10.0 for x in range(0, 10)])
    plt.xlim([0, 1])
    plt.grid()
    plt.show()

    values = df_report['precision'][:-3]
    x_axis = range(len(values))
    threshold = 0.5
    above_threshold = np.maximum(values - threshold, 0)
    below_threshold = np.minimum(values, threshold)

    plt.figure(figsize=(8, 8))
    plt.bar(x_axis, below_threshold, 0.35, color="darkblue")
    plt.bar(x_axis, above_threshold, 0.35, color="green",
            bottom=below_threshold)

    # horizontal line indicating the threshold
    plt.plot([-.5, 2], [threshold, threshold], "k--")

    plt.xticks(ticks=np.arange(len(df_report.index[:-3])), labels=df_report.index[:-3], rotation=45)
    plt.xlabel('Class Label', fontsize=16)
    plt.show()

    return df_report


vgg16 = pd.read_csv('data/models_predictions_zoom_augmentation_block4and5_vgg16_pred_ind.csv', index_col=0)

prediction_result(vgg16, 0.04, confusion_mtx=True)
