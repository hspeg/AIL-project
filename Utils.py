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
from sklearn.model_selection import GridSearchCV

def evaluation_metrics(model, x_train, x_test, y_train, y_test):
    # prediction:
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_proba_test = model.predict_proba(x_test)
    # accuracy and sensitivity
    accuracy = accuracy_score(y_test, y_pred_test)
    sensitivity = recall_score(y_test, y_pred_test)
    # classification reports:
    print("TRAIN SET REPORT:")
    names = ["surv. status >= 5 years", "surv. status < 5 years"]
    print(classification_report(y_train, y_pred_train, target_names=names))
    #
    print("TEST SET REPORT (PREDICTION):")
    c_report_pred = classification_report(y_test, y_pred_test, target_names=names)
    print(c_report_pred)
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
    
    return c_report_pred, auc_m

def cv_scores(model, param_grid, score, x_train, y_train):
    # cross-validation
    grid = GridSearchCV(model, param_grid, scoring=score, cv=5)
    grid.fit(x_train, y_train)
    # plotting the scores
    scores = np.array(grid.cv_results_["mean_test_score"])
    param = np.array([value for value in param_grid.values()])[0]
    param_name = [key for key in param_grid.keys()][0]
    plt.figure()
    plt.plot(param, scores, "o-")
    plt.xlabel(param_name); plt.ylabel(score)
    plt.title("Cross-validation score")
    plt.show()
