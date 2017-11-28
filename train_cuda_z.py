#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:25:22 2017

@author: zhonghao
"""
import sys #make it work on pytorch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from googleNetClass_cuda_z import GoogleNetCuda
import argparse
import pickle
import csv
import copy

classNum = 5
batchSize = 50
numOfElementsPerClass = int(batchSize / classNum)
epochNum = 1000
testNum = 1000
z_height=16
z_width=8

'''
Best practice of using cuda for training
'''
parser = argparse.ArgumentParser(description='train_cuda')
parser.add_argument('--no-cuda', action='store_true', default=False,
help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
'''
description: automatically form a batch for training. Evenly divide image number
             for each class.
input: classZ: number of data per class
       z_value: z_values for all z values
       epochIdx: current epoch idx
output: batch: a batch of z values for for training, [batchSize, (Height+maskHeight-1), (Width+maskWidth-1)]
        truth: 1D array, [classIndex]
side effect: None
'''
def formBatch(classZ, z_value, epochIdx):
    #Zhonghao's first commit forms batch size of 28*28
    startPoint =[]
    truth = []
    for i in range(classNum):
        startPoint.append((epochIdx * numOfElementsPerClass) % classZ[i])
    batch = np.ndarray(shape = (batchSize, len(z_value[0][0])), dtype = float)
    for i in range(classNum):
        for j in range(numOfElementsPerClass):
            #print(z_value[i][(startPoint[i] + j) % classZ[i]])
            batch[i*numOfElementsPerClass+j] = z_value[i][(startPoint[i] + j) % classZ[i]]
            truth.append(i)
    return [batch, truth]

def formTestBatch(z_value_tests, epochIdx):
    truth = []
    curr_idx=epochIdx*numOfElementsPerClass
    #print(curr_idx)
    batch = np.ndarray(shape = (batchSize, len(z_value_tests[0][0])), dtype = float)
    for i in range(classNum):
        for j in range(numOfElementsPerClass):
            batch[i*numOfElementsPerClass+j] = z_value_tests[i][curr_idx+j]
            truth.append(i)
    return [batch, truth]

'''
description: training process. Update weights with each batch of images
input:  batch: from formBatch
        truth: from formBatch
        net: from class Net()
        lossCriterion: what kind of loss function
        optimizer: specify update rules
output: None
side effects:   weights of CNN are updated
'''
def train(batch, truth, net, lossCriterion, optimizer):
    #reshape a batch to [batch size, channels, Height, Width]
    #batchResize = batch.reshape((batchSize, 1, 32, 32))
    batchResize = batch.reshape((batchSize, 1, z_height, z_width))#???input size of z_value
    #convert to a tensor
    batchFeed = Variable(torch.from_numpy(batchResize))
    #inference and back propagate and update weights
    output = net(batchFeed).cuda()
    #print(output.data.numpy()[0], truth[0])
    target = torch.cuda.LongTensor(truth)
    target = Variable(target)
    #print(target)
    optimizer.zero_grad()   # zero the gradient buffers
    loss = lossCriterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()    # Does the update

'''
description: testing process.
input:  batch: from formBatch
        truth: from formBatch
        net: from class Net()

output: return the number of correct predictions within a batch
side effects: None
'''
def test(batch, truth, net):
    #reshape a batch to [batch size, channels, Height, Width]
    #batchResize = batch.reshape((batchSize, 1, 32, 32))
    batchResize = batch.reshape((batchSize, 1, z_height, z_width))
    #convert to a tensor
    batchFeed = Variable(torch.from_numpy(batchResize))
    #inference and back propagate and update weights
    output = net(batchFeed).cuda()
    #output = output.data.numpy()
    _, predicted = torch.max(output.data, 1)
    return (predicted == torch.cuda.LongTensor(truth)).sum()

def get_z_values(classZ,z_value,z_value_tests,filename):
    print(z_value,z_value_tests)
    for i in range(classNum):
        test=0;
        fileread = open('class_z_value/'+filename[i], 'r')
        csvreader = csv.reader(fileread, delimiter=',')
        for row in csvreader:
            oneStroke = []
            for item in row:
                oneStroke.append(float(item))
            if test<testNum:
                z_value_tests[i].append(copy.deepcopy(oneStroke))
                test+=1;
            else:
                z_value[i].append(copy.deepcopy(oneStroke))
                classZ[i]+=1;

def main():
    #initialize
    z_value=[]
    classZ=[0]*classNum;
    z_value_tests=[]

    for i in range(classNum):
        z_value.append(list())
        z_value_tests.append(list())

    #train process
    lossCriterion = nn.CrossEntropyLoss() #using cross entropy loss
    #net = Net()

    p_model = open("googlenet_model.p", 'wb')
    filenames=["eye.csv","finger.csv","foot.csv","hand.csv","leg.csv"];

    if args.cuda:
        net=GoogleNetCuda().double().cuda()
        print("cuda googlenet")
    # create optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # use SGD update rules

    get_z_values(classZ,z_value,z_value_tests,filenames)
    print(classZ,len(z_value_tests))

    for epochIdx in range(epochNum):
        print(epochIdx);
        batch, truth = formBatch(classZ, z_value, epochIdx)
        train(batch, truth, net, lossCriterion, optimizer)
    torch.save(net.state_dict(),p_model)
    print("saved",net.parameters())
    p_model.close()


    '''p_model = open("googlenet_model.p", 'rb')
    stored_net=GoogleNetCuda().double().cuda()
    stored_net.load_state_dict(torch.load(p_model));
    print("reloaded", stored_net.parameters())
    p_model.close()'''
    #test process, still in progress

    correctNum = 0
    for epochIdx in range(int(testNum / batchSize)):
        batch, truth = formTestBatch(z_value_tests, epochIdx)
        correctNum += test(batch, truth, net)
    print("accuracy = %f" %(float(correctNum) / testNum))


if __name__ == "__main__":
    main()
