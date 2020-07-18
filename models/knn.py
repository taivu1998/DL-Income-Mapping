import os, sys

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
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

def perform_knn(dataset, args):
    X_train, X_test, y_train, y_test = dataset
    n_neighbors = 10
    knn = KNeighborsRegressor(n_neighbors, weights = 'uniform')
    knn.fit(X_train, y_train)
    return compute_metrics(knn, X_test, y_test)
