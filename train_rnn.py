#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:53:49 2017

@author: zhonghao
"""

from binary_file_parser import unpack_drawings
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from rdp import rdp

'''
description: input an 2D array, show image of that array on console
input: arr: a numpy 2D array
output: None
side effect: show image on console
'''
def drawStroke(arr):
    a = np.array(arr)
    for i in range(a.shape[0]):
        plt.plot(a[i][0], a[i][1])
    #plt.axis([-5, 260, -5, 260])
    plt.gca().invert_yaxis()
    plt.show()

    
def rawLoad(file):   
    with open(file) as f:
        for line in f:
            j_content = json.loads(line)['drawing']
            drawStroke(j_content)
            
            #print(j_content)
            drawStroke(raw2simplified(j_content))
            
def raw2simplified(arr):
    '''
    a = arr
    if type(arr) is not np.ndarray:
        a = np.array(arr)
    if a.shape[1] == 3:
        a = np.delete(a, 2, 1)
    print(a)
    '''
    Min = [99999999, 99999999] #[x, y]
    Max = [-99999999, -99999999]
    for i in range(len(arr)):
        for j in range(2):
            for k in range(len(arr[i][j])):
                if arr[i][j][k] < Min[j]:
                    Min[j] = arr[i][j][k]
                if arr[i][j][k] > Max[j]:
                    Max[j] = arr[i][j][k]
    scale = max(Max[0] - Min[0], Max[1] - Min[1])
    for i in range(len(arr)):
        for j in range(2):
            for k in range(len(arr[i][j])):
                arr[i][j][k] = int((arr[i][j][k] - Min[j]) * 255 / scale)
    simplifiedArr = []     
    for i in range(len(arr)):
        a = np.zeros([len(arr[i][0]), 2], dtype = int)
        for j in range(len(arr[i][0])):
            a[j][0] = arr[i][0][j]
            a[j][1] = arr[i][1][j]
        a = rdp(a, epsilon=2.0)
        a = np.rollaxis(a, 1)
        #print(a)
        simplifiedArr.append(a.tolist())
    return simplifiedArr
    #b = np.rollaxis(a[0], 1)
    '''       
    drawStroke(arr)
    print(arr)
    '''
    
def simplified2bitmap(arr):
    #render image to 28x28
    pass
    
def main():
    
    rawLoad('leg.ndjson')
    '''
    for d in unpack_drawings('image_bin/leg.bin'):
        drawStroke(d['image'])
    '''
    
def test():
    a = [[1, 2, 3], [4, 5, 6]]
    a = np.array(a)
    if len(a[0].shape) == 1:
        a = np.delete(a, 1, 1)
    print(a)
    #drawStroke(a)
    
if __name__ == "__main__":
    #test()
    main()