#!/usr/bin/env python
# coding: utf-8

# autoencoder: https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798 \
# autoencoder: https://github.com/ardendertat/Applied-Deep-Learning-with-Keras/blob/master/notebooks/Part%203%20-%20Autoencoders.ipynb \
# Unet: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5 \
# Convolutional autoencoder maths: https://pgaleone.eu/neural-networks/2016/11/24/convolutional-autoencoders/ \
# Convolutional autoencoder code: https://blog.keras.io/building-autoencoders-in-keras.html

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import os, os.path
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from IPython.display import clear_output
from keras.utils.vis_utils import plot_model
from keras.layers import Dropout
import math


# In[2]:


def makeLines(array):
#     lineStart=5
#     lineEnd=18
#     arrayLines=np.copy(array)
#     for i in range(lineStart, lineEnd):
#         arrayLines[i] = np.full_like(array[i], np.nan)
#     return arrayLines
    specificLinesIndexes = np.array([2,5,6,9,10,11,14,17,19])
    arrayLines=np.copy(array)
    for index in specificLinesIndexes:
        arrayLines[index] = np.full_like(array[index], np.nan)
    return arrayLines


# In[3]:


numberOfFiles=0
try: 
    filesXtrain = os.listdir('../TrainingDataset/x_train/'); 
    numberOfFiles = len(filesXtrain)
except: print('File not found')
numberOfFiles-=1
testingSetSize = 100
mx_train_lines = np.empty((numberOfFiles-testingSetSize,24,144))
mx_train = np.empty((numberOfFiles-testingSetSize,24,144))
mx_train_lines_nan = np.empty((numberOfFiles-testingSetSize,24,144))
mx_test_lines = np.empty((testingSetSize,24,144))
mx_test_lines_nan = np.empty((testingSetSize,24,144))
mx_test = np.empty((testingSetSize,24,144))
mx_test_infos = np.empty((testingSetSize,7), dtype="O")
mx_test_base = np.empty((testingSetSize,24,144))
allArrays = []
for i in range(numberOfFiles):
    allArrays.append(np.load('../TrainingDataset/x_train/Y2_{}.npy'.format(i), allow_pickle=True, encoding="latin1"))
allArrays = np.asarray(allArrays)

for i in range(0,numberOfFiles-testingSetSize):
    mx_train[i] = allArrays[i][1]
    mx_train_lines[i] = makeLines(mx_train[i])
    mx_train_lines_nan[i] = makeLines(mx_train[i])
for i in range(numberOfFiles-testingSetSize,numberOfFiles):
    mx_test[i-(numberOfFiles-testingSetSize)-1] = allArrays[i][1]
    mx_test_lines[i-(numberOfFiles-testingSetSize)-1] = makeLines(mx_test[i-(numberOfFiles-testingSetSize)-1])
    mx_test_lines_nan[i-(numberOfFiles-testingSetSize)-1] = makeLines(mx_test[i-(numberOfFiles-testingSetSize)-1])
    mx_test_infos[i-(numberOfFiles-testingSetSize)-1] = allArrays[i][2]
    mx_test_base[i-(numberOfFiles-testingSetSize)-1] = allArrays[i][0]

mx_train_lines=np.nan_to_num(mx_train_lines)
mx_train=np.nan_to_num(mx_train)
mx_test_lines=np.nan_to_num(mx_test_lines)
mx_test=np.nan_to_num(mx_test)


# In[4]:


conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([24, 144, 1], input_shape=[24, 144], name="1"),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="relu", name="2_First_convolution"),
    keras.layers.Dropout(0.001, name="3_First_Dropout"),
    keras.layers.MaxPool2D(pool_size=2, name="4_First_Max_Pooling"),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="relu", name="5_Second_Convolution"),
    keras.layers.MaxPool2D(pool_size=2, name="6_Second_Max_Pooling"),
    keras.layers.Conv2D(128, kernel_size=3, padding="SAME", activation="relu", name="7_Third_Convolution"),
])
conv_decoder = keras.models.Sequential([
    keras.layers.Dense(128, input_shape=[6, 36, 128], name="1_Neural_Layer_128"),
    keras.layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding="VALID", activation="relu", name="2_First_Conv2DTranspose"),
    keras.layers.Conv2DTranspose(64, kernel_size=1, strides=1, padding="SAME", activation="relu", name="3_Second_Conv2DTranspose"),
    keras.layers.Dropout(0.001, name="4_First_Dropout"),
    keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="SAME", activation="relu", name="5_Third_Conv2DTranspose"),
    keras.layers.Conv2DTranspose(32, kernel_size=1, strides=1, padding="SAME", activation="relu", name="6_Fourth_Conv2DTranspose"),
    keras.layers.Conv2DTranspose(1, kernel_size=1, strides=1, padding="SAME", activation="sigmoid", name="7_Fifth_Conv2DTranspose"),
    keras.layers.Reshape([24, 144], name="8_Reshape_Output")
])
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])


# In[33]:


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        numberOfPlots = 10
        testPlotsIndex = []
        for i in range(numberOfPlots):
            while(True):
                rand = np.random.randint(0, mx_test_base.shape[0]-10)
                if rand not in testPlotsIndex:
                    break
            testPlotsIndex.append(rand)        
        testPlotsIndex = np.asarray(testPlotsIndex)    
        m=0
        for y in testPlotsIndex:
            prediction = conv_ae.predict(np.expand_dims(mx_test_lines[y],0)).reshape(24,144)
            overlayedTruth = mx_test_lines[y].copy()
            for i in range(mx_test_lines_nan[y].shape[0]):
                if math.isnan(np.sum(mx_test_lines_nan[y][i])):
                    overlayedTruth[i] = prediction[i]

            finalArray = np.array([mx_test_base[y], mx_test[y], mx_test_lines_nan[y], overlayedTruth, mx_test_infos[y]])
            np.save("AE_training_results/AE_training_epoch{}_matrix{}".format(epoch, m), finalArray)
            m+=1


# In[34]:


opt = keras.optimizers.Adam(learning_rate=0.001)
conv_ae.compile(loss="mse", optimizer=opt)
conv_ae.fit(mx_train_lines, mx_train, epochs=20, batch_size=64, shuffle=True, callbacks=[CustomCallback()])

