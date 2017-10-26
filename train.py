#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:25:22 2017

@author: zhonghao
"""

import numpy as np
import matplotlib.pyplot as plt


classNum = 5
batchSize = 50
numOfElementsPerClass = int(batchSize / classNum)
epochNum = 10

'''
description: input an 2D array, show image of that array on console
input: arr: a numpy 2D array
output: None
side effect: show image on console
'''
def imgshow(arr):
    arr_copy = np.copy(arr).reshape(28, 28)
    plt.imshow(arr_copy)
    plt.show()

'''
description: imgLoad loads numpy array of images into an array img
input: img: an empty array []
       imgInfo: an empty array []
output: None
sideffect: change img: images for classNum classes, a 3D array, [class, imageIdx, pixels]
           change imgInfo: [[img number], [category names], img size]
'''
def imgLoad(img, imgInfo):
    imgEye = np.load('image_npy/eye.npy') #125888 imgs
    imgFoot = np.load('image_npy/foot.npy') #203086 imgs
    imgFinger = np.load('image_npy/finger.npy') #167957 imgs
    imgHand = np.load('image_npy/hand.npy') #291773 imgs
    imgLeg = np.load('image_npy/leg.npy') #116804 imgs
    img += [imgEye, imgFoot, imgFinger, imgHand, imgLeg]
    imgInfo += [[imgEye.shape[0], imgFoot.shape[0], imgFinger.shape[0], imgHand.shape[0], imgLeg.shape[0]], 
                    ['eye', 'foot', 'finger', 'hand', 'leg'],
                    imgEye.shape[1]]

'''
description: automatically form a batch for training. Evenly divide image number
             for each class. 
input: img: from imgLoad()
       imgInfo: from imgLoad()
       epochIdx: epoch index
output: batch: a batch of images for training, [image number, pixels]
        truth: 1D array, corresponding categories of images
side effect: None
'''
def formBatch(img, imgInfo, epochIdx):
    startPoint = []
    truth = []
    for i in range(classNum):
        startPoint.append((epochIdx * numOfElementsPerClass) % imgInfo[0][i])
    batch = np.ndarray(shape = (batchSize, imgInfo[2]), dtype = float)
    for i in range(classNum):
        for j in range(numOfElementsPerClass):
            batch[i*numOfElementsPerClass+j] = img[i][(startPoint[i] + j) % imgInfo[0][i]]
            truth.append(imgInfo[1][i])
    return [batch, truth]

'''
description: training process
'''
def train(batch, truth):
    pass

def main():   
    epochIdx = 0
    img = []
    imgInfo = [] #[[img number], [category names], img size]
    imgLoad(img, imgInfo)
    for epochIdx in range(epochNum):
        [batch, truth] = formBatch(img, imgInfo, epochIdx)
        train(batch, truth)
        
if __name__ == "__main__":
    main()
    
    
    