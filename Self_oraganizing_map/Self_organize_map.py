#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:17:27 2019

@author: djinn
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#Reading the dataset
dataset = pd.read_csv("/home/djinn/Desktop/Dl/Self_organizing_map/Self_Organizing_Maps/Self_Organizing_Maps/Credit_Card_Applications.csv")
#print(dataset)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

#TRainig The Som
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len= 15, sigma=1.0,learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100)

#Visualizing The Results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,customer in enumerate(x):
    w = som.winner(customer)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markeredgewidth = 2)
show()

#Finding The Frauds
#SOME ERROR IN BELOW LINES DIMESION ERROR
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)




