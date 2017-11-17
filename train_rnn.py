#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:53:49 2017

@author: zhonghao
"""

from binary_file_parser import unpack_drawings
import numpy as np
import matplotlib.pyplot as plt

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
    plt.axis([-5, 260, -5, 260])
    plt.gca().invert_yaxis()
    
    plt.show()
    

    
def main():
    for d in unpack_drawings('image_bin/leg.bin'):
        drawStroke(d['image'])
        print(d['image'])
    
def test():
    a = [[[1, 2, 3], [4, 5, 6]]]
    drawStroke(a)
    
if __name__ == "__main__":
    #test()
    main()