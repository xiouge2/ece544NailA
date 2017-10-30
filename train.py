#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:25:22 2017

@author: zhonghao
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

classNum = 5
batchSize = 50
numOfElementsPerClass = int(batchSize / classNum)
epochNum = 1000
testNum = 100

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
output: batch: a batch of images for training, [batchSize, (Height+maskHeight-1), (Width+maskWidth-1)]
        truth: 2D array, [image index, truth vector]
               truth vector: For example, if eye is true, then truth vector is [1, 0, 0, 0, 0]
side effect: None
'''
def formBatch(img, imgInfo, epochIdx):
    startPoint = []
    truth = []
    for i in range(classNum):
        startPoint.append((epochIdx * numOfElementsPerClass) % imgInfo[0][i])
    batch = np.zeros((batchSize, 32, 32), dtype = np.float32)
    for i in range(classNum):
        for j in range(numOfElementsPerClass):
            # add halograms to an image, mask size is 5x5
            batch[i*numOfElementsPerClass+j][2:30, 2:30] = (img[i][(startPoint[i] + j) % imgInfo[0][i]]).reshape((28, 28))
            truth.append(i)
    return batch, truth

'''
description: automatically form a batch for testing. Evenly divide image number
             for each class. 
input: img: from imgLoad()
       imgInfo: from imgLoad()
       epochIdx: epoch index
output: batch: a batch of images for training, [batchSize, (Height+maskHeight-1), (Width+maskWidth-1)]
        truth: 2D array, [image index, truth vector]
               truth vector: For example, if eye is true, then truth vector is [1, 0, 0, 0, 0]
side effect: None
'''
def testBatch(img, imgInfo, epochIdx):
    startPoint = []
    truth = []
    for i in range(classNum):
        startPoint.append((epochIdx * numOfElementsPerClass) % imgInfo[0][i])
    batch = np.zeros((batchSize, 32, 32), dtype = np.float32)
    for i in range(classNum):
        for j in range(numOfElementsPerClass):
            # add halograms to an image, mask size is 5x5
            batch[i*numOfElementsPerClass+j][2:30, 2:30] = (img[i][(startPoint[i] + j) % imgInfo[0][i]]).reshape((28, 28))
            truth.append(i)
    return batch, truth

'''
description: Now we are using LeNet. We will change to GoogLeNet later.
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

'''
description: training process. Update weights with each batch of images
input:  batch: from formBatch
        truth: from formBatch
        net: from class Net()
        lossCriterion: what kind of loss function
        optimizer: specify update rules
output: None
side effects:   weights of CNN is updated
'''
def train(batch, truth, net, lossCriterion, optimizer):
    #reshape a batch to [batch size, channels, Height, Width]
    batchResize = batch.reshape((batchSize, 1, 32, 32))
    #convert to a tensor
    batchFeed = Variable(torch.from_numpy(batchResize))
    #inference and back propagate and update weights
    output = net(batchFeed)
    #print(output.data.numpy()[0], truth[0])
    target = torch.LongTensor(truth)
    target = Variable(target)
    optimizer.zero_grad()   # zero the gradient buffers
    loss = lossCriterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

'''
description: testing process. 
input:  batch: from formBatch
        truth: from formBatch
        net: from class Net()

output: return the correct instances within a batch
side effects: None
'''
def test(batch, truth, net):
    #reshape a batch to [batch size, channels, Height, Width]
    batchResize = batch.reshape((batchSize, 1, 32, 32))
    #convert to a tensor
    batchFeed = Variable(torch.from_numpy(batchResize))
    #inference and back propagate and update weights
    output = net(batchFeed)
    #output = output.data.numpy()
    _, predicted = torch.max(output.data, 1)
    return (predicted == torch.LongTensor(truth)).sum()
    
    

def main():  
    # load process
    img = []
    imgInfo = [] #[[img number], [category names], img size]
    imgLoad(img, imgInfo)
    
    #train process
    lossCriterion = nn.CrossEntropyLoss() #using cross entropy loss
    net = Net()
    # create optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # use SGD update rules
    for epochIdx in range(epochNum):
        batch, truth = formBatch(img, imgInfo, epochIdx)
        train(batch, truth, net, lossCriterion, optimizer)
        
    #test process, still in progress
    correctNum = 0
    for epochIdx in range(int(testNum / batchSize)):
        batch, truth = formBatch(img, imgInfo, epochIdx)
        correctNum += test(batch, truth, net)
    print("accuracy = %f" %(float(correctNum) / testNum))
        
    

        
if __name__ == "__main__":
    main()
    
    
    