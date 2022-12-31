# Some ai and ml models evaluation functions

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

def evaluation_metrics(model, x_train, x_test, y_train, y_test):
    # prediction:
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_proba_test = model.predict_proba(x_test)
    # classification reports:
    print("TRAIN SET REPORT:")
    names = ["surv. status >= 5 years", "surv. status < 5 years"]
    print(classification_report(y_train, y_pred_train, target_names=names))
    #
    print("TEST SET REPORT:")
    print(classification_report(y_test, y_pred_test, target_names=names))
    # ROC curve and AUC
    fpr, tpr, treshold = roc_curve(y_test, y_pred_proba_test[:,1])
    auc_m = auc(fpr, tpr)
    # plotting
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, "b", label="AUC: "+str(round(auc_m, 3)))
    plt.legend(loc='lower right')
    plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.title("ROC")
    plt.xlabel("False positive rate"), plt.ylabel("True positive rate")
    plt.show() 


