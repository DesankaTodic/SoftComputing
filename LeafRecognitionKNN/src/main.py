#link ka dataset-u: http://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/
#racunanja rastojanja: http://dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
#osobine regionprops-a: http://www-rohan.sdsu.edu/doc/matlab/toolbox/images/regionprops.html#254911

import matplotlib.pyplot as plt  # za prikaz slika, grafika, itd.
import numpy as np
from numpy import zeros
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import square
from skimage.morphology import opening, closing
import collections
import datetime
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.measure import label  # implementacija connected-components labelling postupka
from skimage.measure import regionprops
from skimage.morphology import square, diamond, disk  # strukturni elementi
from pylab import *
from resizeimage import resizeimage
import csv
import random
import math
import operator
"""
def processRegions(path,leafNum):
    img = imread(path)
    image_gray = rgb2gray(img)  # transformacija u nijanse sive
    image_bw = (image_gray > 0.8).astype('uint8')  # transformacija u crno bijelu sliku
    #plt.imshow(image_bw, 'gray')
    # plt.show()
    img_tr_dil = dilation(image_bw, selem=diamond(3))
    #plt.imshow(img_tr_dil, 'gray')
    # plt.show()
    img_tr_er = erosion(img_tr_dil, selem=diamond(3))
    #plt.imshow(img_tr_er, 'gray')
    # plt.show()
    labeled_img = label(img_tr_er)  # rezultat je slika sa obelezenim regionima
    #plt.imshow(labeled_img, 'gray')
    # plt.show()
    regions = regionprops(labeled_img)
    #count = 0
    line = ''
    for region in regions:
        bbox = region.bbox
        height = bbox[2] - bbox[0]  # visina
        width = bbox[3] - bbox[1]  # sirina
        if height > 50 or width > 50:
            #area - Returns a scalar that specifies the actual number of pixels in the region.
            #major_axis_length/minor_axis_length - Returns a scalar that specifies the length (in pixels) of the major/minor axis of the ellipse that has
            # the same normalized second central moments as the region
            normalArea = region.area/1000 #skaliranje povrsine posto je 1000 puta veca od osa, da bi isto doprinosile prilikom racunanja rastojanja
            # line = str(round(region.area, 2)) + ',' + str(round(region.major_axis_length, 2))  + ',' + leafNum
            line = str(round(normalArea, 2))  + ',' + str(round(region.major_axis_length, 2))  + ',' + str(round(region.minor_axis_length, 2)) + ',' + leafNum
    return line
path1 = 'train_images/leaf1/l1nr001.tif'
path2 = 'train_images/leaf2/l2nr001.tif'
path3 = 'train_images/leaf3/l3nr001.tif'
path4 = 'train_images/leaf4/l4nr001.tif'
path5 = 'train_images/leaf5/l5nr001.tif'
path6 = 'train_images/leaf6/l6nr001.tif'
path7 = 'train_images/leaf7/l7nr001.tif'
path8 = 'train_images/leaf8/l8nr001.tif'
path9 = 'train_images/leaf9/l9nr001.tif'
path10 = 'train_images/leaf10/l10nr001.tif'
path11 = 'train_images/leaf11/l11nr001.tif'
path12 = 'train_images/leaf12/l12nr001.tif'
path13 = 'train_images/leaf13/l13nr001.tif'
path14 = 'train_images/leaf14/l14nr001.tif'
path15 = 'train_images/leaf15/l15nr001.tif'
#otvaranje fajla za cuvanje listova oznacenih po atributima u rezimu pisanja
file = open('leaf.data','w')

def processPathsToImages(path,imageRootName):
    for xxx in range(1, 76):
        splits = path.split('/')
        if xxx<10:
            splits[2] = imageRootName + str(xxx) + '.tif'
        else:
            if splits[1]=='leaf10' or splits[1]=='leaf11' or splits[1]=='leaf12' or splits[1]=='leaf13' or splits[1]=='leaf14' or splits[1]=='leaf15':
                splits[2] =  imageRootName[0:6] + str(xxx) + '.tif'
            else:
                splits[2] = imageRootName[0:5] + str(xxx) + '.tif'
        # print splits
        path = splits[0] + '/' + splits[1] + '/' + splits[2]
        #print path
        line = processRegions(path, imageRootName)
        file.write(line + '\n')

print "Begin :"
print  datetime.datetime.now().time()
#processPathsToImages(path1,'l1nr00')
processPathsToImages(path2,'l2nr00')
processPathsToImages(path3,'l3nr00')
processPathsToImages(path4,'l4nr00')
processPathsToImages(path5,'l5nr00')
processPathsToImages(path6,'l6nr00')
processPathsToImages(path7,'l7nr00')
processPathsToImages(path8,'l8nr00')
processPathsToImages(path9,'l9nr00')
processPathsToImages(path10,'l10nr00')
processPathsToImages(path11,'l11nr00')
processPathsToImages(path12,'l12nr00')
processPathsToImages(path13,'l13nr00')
processPathsToImages(path14,'l14nr00')
#processPathsToImages(path15,'l15nr00')
file.close()
"""
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):  # da od stringa upisanog u fajl napravi float kad su posmatrana 3 parametra lista
            #for y in range(2):  # da od stringa upisanog u fajl napravi float kad su posmatrana 2 parametra lista
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):#euklidsko rastojanje
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance +=abs(instance1[x] - instance2[x])
    return distance

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1 #da ne uzima u obzir iza posljednje zapete, tj. naziv klase
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        #dist = manhattanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('leaf.data', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

main()
print "End :"
print  datetime.datetime.now().time()
