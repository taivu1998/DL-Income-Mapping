import os, sys
import time
#import math
#import json
import numpy as np
#import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
#from torch.autograd import Variable
import torch.utils.data as utils
# import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
#import torchvision.datasets as datasets

from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import util

import warnings
warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self, image_size):
        super(Net, self).__init__()
        if image_size == 256:
            LINEAR_INPUT_SIZE_FC1 = 64 * 9 * 9
        elif image_size == 224:
            LINEAR_INPUT_SIZE_FC1 = 64 * 8 * 8
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size = 7, padding = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(LINEAR_INPUT_SIZE_FC1, 256),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze(0)


def train(epoch, net, criterion, optimizer, trainLoader, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        util.progress_bar(batch_idx, len(trainLoader),
                          'Train Loss: %.3f' % (train_loss / (batch_idx + 1)))


def test(epoch, net, criterion, testLoader, device):
    net.eval()
    test_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(testLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        util.progress_bar(batch_idx, len(testLoader),
                          'Test Loss: %.3f' % (test_loss / (batch_idx + 1)))
        
    return test_loss / len(testLoader)
        

def save_checkpoint(net, optimizer, loss, epoch):
    print("==> Saving checkpoint..")
    state = {
        'state_dict': net.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth.tar')


def load_checkpoint():
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth.tar')
    net = checkpoint['net']
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch'] + 1
    optimizer = checkpoint['optimizer']
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
    return net, best_loss, start_epoch, optimizer


def compute_metrics(net, X_test, y_test):
    net.eval()
    y_pred = net(X_test)
    y_pred = y_pred.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'mean_squared_error' : mse,
        'mean_absolute_error' : mae,
        'r2_score' : r2
    }
    return metrics


def perform_cnn(dataset, args):
    if args.seed != 0:
        torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    best_loss = float('inf')
        
    X_train, X_test, y_train, y_test = dataset
    trainSet = utils.TensorDataset(X_train, y_train)
    testSet = utils.TensorDataset(X_test, y_test)
    trainLoader = utils.DataLoader(trainSet, shuffle = True,
                                   batch_size = args.batch_size)
    testLoader = utils.DataLoader(testSet, shuffle = False,
                                  batch_size = args.batch_size)
    
    if args.metric == 'mean_squared_error':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    
    if args.arch is None:
        image_size = 224 if args.augment else 256
        net = Net(image_size)
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr = args.lr,
                               weight_decay = args.decay)
        
    elif args.arch == 'vgg16':
        net = torchvision.models.vgg16(pretrained = True)
        for param in net.parameters():
            param.requires_grad = False
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1), nn.ReLU())
        net = net.to(device)
        optimizer = optim.Adam(net.classifier[6].parameters(),
                               lr = args.lr, weight_decay = args.decay)
        
    else:
        resnets = {
            'resnet18' : models.resnet18,
            'resnet34' : models.resnet34,
            'resnet50' : models.resnet50,
            'resnet101' : models.resnet101,
            'resnet152' : models.resnet152,
        }
        net = resnets[args.arch](pretrained = True)
        for param in net.parameters():
            param.requires_grad = False
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.ReLU())
        net = net.to(device)
        optimizer = optim.Adam(net.fc.parameters(),
                               lr = args.lr, weight_decay = args.decay)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma = 0.1,
                                             step_size = 10)
    
    if args.resume:
        net, best_loss, start_epoch, optimizer = load_checkpoint()
    
    for epoch in range(args.start_epoch, args.epochs):
        train(epoch, net, criterion, optimizer, trainLoader, device)
        lr_scheduler.step()
        test_loss = test(epoch, net, criterion, testLoader, device)
        if test_loss < best_loss:
            best_loss = test_loss
            save_checkpoint(net, optimizer, best_loss, epoch)

    return compute_metrics(net, X_test, y_test)
