import numpy as np
from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_curve, roc_auc_score)


def get_metric(y_true, y_pred, y_prob):

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    Auc = auc(fpr, tpr)

    precision1, recall1, _ = precision_recall_curve(y_true, y_prob)
    Aupr = auc(recall1, precision1)

    return Auc, Aupr, accuracy, precision, recall, f1, mcc