import os, sys

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'mean_squared_error' : mse,
        'mean_absolute_error' : mae,
        'r2_score' : r2
    }
    return metrics

def perform_svm(dataset, args):
    X_train, X_test, y_train, y_test = dataset
    if args.kernel == 'rbf':
        svm = SVR(kernel = 'rbf', C = 100, gamma = 0.1, epsilon = .1)
    elif args.kernel == 'linear':
        svm = SVR(kernel = 'linear', C = 100, gamma = 'auto')
    elif args.kernel == 'poly':
        svm = SVR(kernel = 'poly', C = 100, gamma = 'auto',
                  degree = 3, epsilon = .1, coef0 = 1)
        
    svm.fit(X_train, y_train)
    return compute_metrics(svm, X_test, y_test)
