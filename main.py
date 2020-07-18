'''
This program implements a machine learning program for predicting income distribution
in California.
'''

import os, sys
sys.path.append('./utils')
sys.path.append('./models')
sys.path.append('./data')
import argparse

from utils import data_processing
from models import cnn, lr, rf, knn, svm

import warnings
warnings.filterwarnings('ignore')


model_options = ['cnn', 'lr', 'rf', 'knn', 'svm']
arch_options = [None, 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16']
regularize_options = [None, 'ridge', 'lasso', 'elastic_net']
kernel_options = ['rbf', 'linear', 'poly']
metric_options = ['mean_squared_error', 'mean_absolute_error', 'r2_score']


def parseArgs():
    ''' Reads command line arguments. '''
    parser = argparse.ArgumentParser(description = 'Mapping Income Distribution with Machine Learning.',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', type = str, default = 'cnn',
                        help = 'Learning models.', choices = model_options)
    parser.add_argument('--arch', type = str, default = None,
                        help = 'CNN architecture.', choices = arch_options)
    parser.add_argument('--metric', type = str, default = 'mean_absolute_error',
                        help = 'Evaluation metric.', choices = metric_options)
    
    # Arguments for training CNN.
    parser.add_argument('--batch-size', type = int, default = 16, help = 'Batch size.')
    parser.add_argument('--lr', type = float, default = 0.1, help = 'Learning rate.')
    parser.add_argument('--start-epoch', type = int, default = 0, help = 'Starting epoch.')
    parser.add_argument('--epochs', type = int, default = 200, help = 'Number of epochs.')
    parser.add_argument('--augment', action = 'store_true', default = False,
                        help = 'Augment data by flipping and cropping (for transfer learning).')
    parser.add_argument('--decay', type = float, default = 1e-4, help = 'Weight decay.')
    parser.add_argument('--momentum', default = 0.9, type = float,
                        metavar = 'M', help = 'Momentum.')
    parser.add_argument('--seed', type = int, default = 0, help = 'Random seed.')
    parser.add_argument('--resume', action = 'store_true', default = False,
                        help = 'Resume from checkpoint.')
    
    # Arguments for machine learning models.
    parser.add_argument('--regularize', type = str, default = None,
                        help = 'Regularization.', choices = regularize_options)
    parser.add_argument('--kernel', type = str, default = 'rbf',
                        help = 'Kernel for SVMs.', choices = kernel_options)
                        
    args = parser.parse_known_args()[0]
    return args


def main():
    ''' Main program. '''
    print("Welcome to Our Machine Learning Program.")
    args = parseArgs()
    print("==> Preparing data..")
    X_train, X_test, y_train, y_test = data_processing.prepareDataset(args.model, args.augment)
    print("==> Training model..")
    
    if args.model == 'cnn':
        metrics = cnn.perform_cnn((X_train, X_test, y_train, y_test), args)
    elif args.model == 'lr':
        metrics = lr.perform_lr((X_train, X_test, y_train, y_test), args)
    elif args.model == 'rf':
        metrics = rf.perform_rf((X_train, X_test, y_train, y_test), args)
    elif args.model == 'knn':
        metrics = knn.perform_knn((X_train, X_test, y_train, y_test), args)
    elif args.model == 'svm':
        metrics = svm.perform_svm((X_train, X_test, y_train, y_test), args)
    
    print()
    print(args.metric, metrics[args.metric])
    
    
if __name__ == '__main__':
    main()
