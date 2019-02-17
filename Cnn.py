#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 23:37:31 2019

@author: djinn
"""

#Part 1  Building The Cnn
#Importing The Keras Libraries and Packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Intialize The Cnn
classifier = Sequential()
classifier.add(Convolution2D( 32, 3, 3,input_shape=( 64, 64, 3), activation='relu' ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D( 32, 3, 3, activation='relu' ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu' ))
classifier.add(Dense(output_dim = 64, activation = 'relu' ))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid' ))
#Compiling
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'] )

#Fitting cnn to the images
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
'/home/djinn/Desktop/Dl/Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/training_set',
target_size=(64, 64),
batch_size=32,
class_mode='binary')

test_set = test_datagen.flow_from_directory(
'/home/djinn/Desktop/Dl/Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/test_set',
target_size=(64, 64),batch_size=32,class_mode='binary')

classifier.fit_generator(
                    training_set,
                    steps_per_epoch=8000,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=2000)

import numpy as np
from keras.preprocessing import image

test_image= image.load_img('/home/djinn/Desktop/Dl/Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_2.jpg',
                           target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
if result[0][0] ==1 :
    predication = "dog"
else:
    predication = "cat"
print(predication)




