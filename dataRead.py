#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:29:34 2018

@author: shymacbook
"""

import pandas as pd

# read and format input data
train_DF = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/hw2_ANN/MNIST_in_csv/mnist_train.csv')

train_DF.head()
train_DF.columns[101:200]

pixels = 28*28
pixArray = []
for x in range(0,pixels + 1):
    if x == 0:
        pixArray.append('target')
    else:
        pixArray.append(x)

train_DF.columns = pixArray
train_DF.head()    


train_DF.target.unique()
train_DF['l0'] = 0
train_DF['l1'] = 0
train_DF['l2'] = 0
train_DF['l3'] = 0
train_DF['l4'] = 0
train_DF['l5'] = 0
train_DF['l6'] = 0
train_DF['l7'] = 0
train_DF['l8'] = 0
train_DF['l9'] = 0

labelArray = ['l0','l1','l2','l3','l4','l5','l6','l7','l8','l9']

# for x in range(0, len(train_DF.index)):
for index, row in train_DF.iterrows():
    print('index = ', index)
    # print('row = ', row)
    # lab = train_DF.label[index]
    lab = row['target']
    # train_DF[index][labelArray[index]] = 1
    row[labelArray[lab]] = 1
    
train_DF.head()
train_DF.iloc[0][-10:]      # row 0, last 10 cols - TARGETS
train_DF.iloc[0][0]         # row 0, col 0
train_DF.iloc[0]            # entire row 0
train_DF.iloc[0][1:-10]     # row 0, cols 1 through 10 from last - INPUTS

import numpy as np
r1 = np.linspace(-0.01,0.01,784)
len(r1)
