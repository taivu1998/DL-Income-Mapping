import os, sys

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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

def perform_lr(dataset, args):
    if args.regularize == 'ridge':
        lr = Ridge(alpha = 1.0)
    elif args.regularize == 'lasso':
        lr = Lasso(alpha = 1.0)
    elif args.regularize == 'elastic_net':
        lr = ElasticNet(alpha = 0.1, l1_ratio = 0.7)
    else:
        lr = LinearRegression()
    
    X_train, X_test, y_train, y_test = dataset
    lr.fit(X_train, y_train)
    return compute_metrics(lr, X_test, y_test)
