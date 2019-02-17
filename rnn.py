#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:36:33 2019

@author: djinn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#TRAINING DATA SET
dataset_train = pd.read_csv("/home/djinn/Desktop/Dl/Recurrent_Neural_Networks-1/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")
trainig_set = dataset_train.iloc[:,1:2].values
print(trainig_set)

#fEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(trainig_set)


##PREPROCESSING
#Creating A data with 60 values with  1 output
x_train = []
y_train = []
for i in range(120,1258):
    x_train.append(training_set_scaled[i-120:i,0])
    y_train.append(training_set_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)
#RESHAPING
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

##BUILDING THE RUN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

Regresser = Sequential()
#FIRST LAYER 
Regresser.add(LSTM(units = 100,return_sequences=True,input_shape =(x_train.shape[1],1)))
Regresser.add(Dropout(rate=0.2))
#SECOND LAYER
Regresser.add(LSTM(units = 100,return_sequences=True))
Regresser.add(Dropout(rate=0.2))

#Extra try layer
Regresser.add(LSTM(units = 100,return_sequences=True))
Regresser.add(Dropout(rate=0.2))
#THIRD LAYER
Regresser.add(LSTM(units = 100,return_sequences=True))
Regresser.add(Dropout(rate=0.2))
#FORTH LAYER
Regresser.add(LSTM(units = 100))
Regresser.add(Dropout(rate=0.2))

#FINAL OUTPUT LAYER
Regresser.add(Dense(units=1))

#COMPILING RNN LAYER
Regresser.compile(optimizer='adam',loss='mean_squared_error')

#FITTING THE RNN TO TRAINIING SET
Regresser.fit(x_train,y_train,epochs=150,batch_size=32)

dataset_test = pd.read_csv("/home/djinn/Desktop/Dl/Recurrent_Neural_Networks-1/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv")
test_price = dataset_test.iloc[:,1:2].values

#GETTING THE PREDICATION STOCK PRICE 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
print(inputs)
inputs = sc.transform(inputs)
x_test = []
for i in range(120,140):
    x_test.append(inputs[i-120:i,0])
print(x_test)
x_test = np.array(x_test)
print(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_price = Regresser.predict(x_test)
predicted_price = sc.inverse_transform(predicted_price)

#VISUALISING
plt.plot(test_price,color = 'red',label ='Real Google Stock Price')
plt.plot(predicted_price,color = 'blue',label ='Predicted Google Stock Price') 
plt.title('Google Stock Price Prdication')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_price,predicted_price))

