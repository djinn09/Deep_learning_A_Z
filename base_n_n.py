#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 00:52:00 2019

@author: djinn
"""

import numpy as np

alphas = [0.01,0.01,0.1,1,10,100,1000]
hiddensize = 32

def sigmoid(x):
    op=1/(1+np.exp(-x))
    return op

def sigmoid_output_to_derivative(output):
    return output*(1-output)

x = np.array([
        [0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1]])
y = np.array([
        [0],
        [1],
        [1],
        [0]])
    
for alpha in alphas:
    print("Trainin with Alpha:",alpha)
    np.random.seed(1)
    syn_0 = 2*np.random.random((3,hiddensize))
    syn_1 = 2*np.random.random((hiddensize,1))
    
    for i in range(60000):
        layer_0=x
        layer_1 = sigmoid(np.dot(layer_0,syn_0))
        layer_2 = sigmoid(np.dot(layer_1,syn_1))
        
        layer_2_error = layer_2 - y
        
        if (i%1000) == 0:
            print("Error After",i, "iteration",np.mean(np.abs(layer_2_error)))
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(syn_1.T)
        layer_1_delta = layer_1_error*sigmoid_output_to_derivative(layer_1)
        
        syn_1 = alpha*(layer_1.T.dot(layer_2_delta))
        syn_0 = alpha*(layer_0.T.dot(layer_1_delta))
        








