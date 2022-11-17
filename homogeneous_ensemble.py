# -*- coding: utf-8 -*-
"""
/**************************************************************************
***************************** Ensemble Method *****************************
***************************************************************************
Program Name         : ensemble.py
Owner                : Jeremy Carew
Requestor            : Department of Mathematics, Tennessee Tech
Approximate Run Time : N/A
Program Description  : This module houses the ensemble method used in the
                       wind power prediction paper 
Input                : N/A
...
Output               : N/A
...
Dependencies         : numpy
                       pandas
                       sklearn.linear_model
                       sklearn.svm
                       sklearn.tree
                       sklearn.metrics
                       random
Macro usage          : N/A
Audit Trail          :
2022-11-16  Jeremy Carew   Created file;
***************************************************************************
***************************************************************************
**************************************************************************/
"""

#  Importing required libraries
import numpy as np
import random
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def homog_ens(data, algorithm, number_of_preds):
    
    #  Classifiers available to the ensemble
    predictors = [DecisionTreeClassifier(max_depth=5),
                  SVR(kernel="rbf", C=10000, tol=1e-5),
                  KNeighborsClassifier(n_neighbors=random.randint(1,25))]    

    weak_preds = []
    weights = []

    for i in range(0,number_of_preds):
        #  Sampling of Data
        train_set, test_set = split_train_test(data, 0.2)
        X_vars = train_set.iloc[:,1:-1]
        X_labels = train_set.iloc[:,-1]
        Y = X_labels.to_numpy()
        X = X_vars.to_numpy()
        #  Model training
        weak_lin_reg = predictors[algorithm]
        weak_lin_reg.fit(X, Y)
        weak_preds.append(weak_lin_reg)
        weights.append(1 / mean_squared_error(weak_lin_reg.predict(test_set.iloc[:,1:-1].to_numpy()), 
                                              test_set.iloc[:,-1].to_numpy()))
    return weights, weak_preds