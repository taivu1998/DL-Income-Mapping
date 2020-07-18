import os, sys
import math
import io
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import util
import webmercator

import warnings
warnings.filterwarnings('ignore')


class Mapping(object):

    MIN_X = 2794
    MAX_X = 2838
    MIN_Y = 6528
    MAX_Y = 6571

    def __init__(self):
        self.convertImagesToNpz(range(self.MIN_X, self.MAX_X + 1),
                                range(self.MIN_Y, self.MAX_Y + 1))
        self.loadImages()
        self.readZipCodesData()
        self.readCensusData()
        
        self.mappingSaved = False
        self.mapping = defaultdict(list)
        self.labels = np.zeros(self.getNumImages())
        self.income = np.zeros(self.getNumImages())
        self.population = np.zeros(self.getNumImages())
    
    def convertImagesToNpz(self, range_x, range_y):
        if not os.path.isdir('./data/imagery_npz'):
            os.mkdir('./data/imagery_npz')
            for x in range_x:
                for y in range_y:
                    jpg = './data/imagery/14_{}_{}.jpg'.format(x, y)
                    npz = './data/imagery_npz/14_{}_{}.npz'.format(x, y)
                    util.jpg_to_npz(jpg, npz)
    
    def getTile(self, coordinates):
        x, y = coordinates
        npz = './data/imagery_npz/14_{}_{}.npz'.format(x, y)
        return np.load(npz)['arr_0']
    
    def loadImages(self):
        result = []
        for x in range(self.MIN_X, self.MAX_X + 1):
            for y in range(self.MIN_Y, self.MAX_Y + 1):
                result.append(self.getTile((x, y)))
        self.images = np.array(result)

    def readZipCodesData(self):
        self.zipCodesData = pd.read_csv('./data/ziplatlon.csv', delimiter = ';',
                                        usecols = ['zip', 'latitude', 'longitude'])
        self.zipCodes = list(self.zipCodesData['zip'])

    def readCensusData(self):
        self.censusData = pd.read_csv('./data/16zpallnoagi.csv',
                                      usecols = ['ZIPCODE', 'N1', 'A02650'])
        zipCodes = set(self.censusData['ZIPCODE'])
        self.zipCodes = list(zipCodes.intersection(self.zipCodes))

    def createMapping(self):
        remainingImages = list(range(self.getNumImages()))
        allTiles = [(x, y) for x in range(self.MIN_X, self.MAX_X + 1) \
                           for y in range(self.MIN_Y, self.MAX_Y + 1)]
        tilesToIndexes = {tile : index for index, tile in enumerate(allTiles)}
        
        zipCodesToTiles = {}
        for i in range(self.getNumZipCodes()):
            z, lat, lon = self.zipCodesData.loc[i]
            z = int(z)
            if z in self.zipCodes:
                x, y = webmercator.xy(lat, lon, z = 14)
                x, y = int(x), int(y)
                zipCodesToTiles[z] = (x, y)
        
        for z in zipCodesToTiles:
            (x, y) = zipCodesToTiles[z]
            if (x, y) in allTiles:
                self.mapping[z].append(tilesToIndexes[(x, y)])
                if tilesToIndexes[(x, y)] in remainingImages:
                    remainingImages.remove(tilesToIndexes[(x, y)])
                    
        for index in remainingImages:
            (currX, currY) = allTiles[index]
            distances = []
            for z in zipCodesToTiles:
                (x, y) = zipCodesToTiles[z]
                distance = (currX - x)**2 + (currY - y)**2
                distances.append((distance, z))
            minZ = min(distances)[1]
            self.mapping[minZ].append(index)
        
        self.mappingSaved = True

    def saveMapping(self):
        with open('./data/mapping.json', 'w') as fp:
            json.dump(self.mapping, fp)

    def loadMapping(self):
        with open('./data/mapping.json') as fp:
            self.mapping = json.load(fp)
        self.mapping = {int(key) : value for key, value in self.mapping.items()}
            
    def createLabels(self):
        for z in self.mapping:
            data = self.censusData[self.censusData['ZIPCODE'] == z]
            totalPopulation = float(data['N1'])
            totalIncome = float(data['A02650'])
            population = totalPopulation / len(self.mapping[z])
            income = totalIncome / len(self.mapping[z])
            for index in self.mapping[z]:
                self.population[index] += population
                self.income[index] += income
        
        for index in range(self.getNumImages()):
            if self.population[index] == 0:
                self.population[index] += 1
        self.labels = self.income / self.population

    def getLabeledData(self):
        return self.images, self.labels

    def getNumImages(self):
        return self.images.shape[0]

    def getNumZipCodes(self):
        return len(self.zipCodes)


def process(X, transform):
    if transform:
        X = [transform(img).unsqueeze(0) for img in X]
        X = torch.cat(X)
    else:
        X = np.array([x.flatten() / 255 for x in X])
    return X

def prepareDataset(model, augment):
    
    mapping = Mapping()
    try:
        mapping.loadMapping()
    except:
        mapping.createMapping()
        mapping.saveMapping()
    mapping.createLabels()
    X, y = mapping.getLabeledData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        random_state = 0)
    
    if model == 'cnn':
        if augment:
            transform_train = transforms.Compose([
                            transforms.Lambda(lambda x: x.astype(np.uint8)),
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ])

            transform_test = transforms.Compose([
                        transforms.Lambda(lambda x: x.astype(np.uint8)),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                    ])
                    
        else:
            transform_train = transforms.Compose([
                            transforms.Lambda(lambda x: x.astype(np.uint8)),
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ])

            transform_test = transforms.Compose([
                        transforms.Lambda(lambda x: x.astype(np.uint8)),
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                    ])
        
        X_train = process(X_train, transform_train)
        y_train = torch.from_numpy(y_train).float()
        X_test = process(X_test, transform_test)
        y_test = torch.from_numpy(y_test).float()
        
    else:
        transform_train = None
        transform_test = None
        X_train = process(X_train, transform_train)
        X_test = process(X_test, transform_test)
    
    return X_train, X_test, y_train, y_test
