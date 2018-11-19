# Python Test client file

import exampleANN as ANN


import pandas as pd
import numpy as np

# read and format input data
train_DF = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/hw2_ANN/MNIST_in_csv/mnist_train.csv')

train_DF.head()

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
    # print('index = ', index)
    # print('row = ', row)
    # lab = train_DF.label[index]
    lab = row['target']
    # train_DF[index][labelArray[index]] = 1
    row[labelArray[lab]] = 1
    
# print(train_DF.head())

# epochs
ep = 1000

# executing test script for ANN
# nn = ANN.NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
# for i in range(ep):
#     nn.train([0.05, 0.1], [0.01, 0.99])
#     print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# testing ANN on row 1 of data frame
numInputs = 784					# 785? or 784? <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ERROR index
numHiddenNeurons = 10
numOutputs = 10
# make correct # of weights
# hiddenWeights = [-0.01, -0.008, -0.006, -0.003, -0.001, 0.001, 0.003, 0.006, 0.008, 0.01]
hiddenWeights = list(np.linspace(-0.01,0.01, (numInputs * numHiddenNeurons)))
hiddenBias = 1
# make correct # of weights
# outputWeights = [-0.01, -0.008, -0.006, -0.003, -0.001, 0.001, 0.003, 0.006, 0.008, 0.01]
outputWeights = list(np.linspace(-0.01,0.01, (numHiddenNeurons * numOutputs)))
outputBias = 1
nnRow1 = ANN.NeuralNetwork(numInputs, numHiddenNeurons, numOutputs, hiddenWeights, hiddenBias, outputWeights, outputBias)
# pick row numbrer
rowIndex = 0
trainOUT = train_DF.iloc[rowIndex][-10:] 		# last 10 columns of targets
trainIN = train_DF.iloc[rowIndex][1:-10]		# pixel input colummns
for j in range(ep):
	nnRow1.train(trainIN, trainOUT)
	print(i, round(nnRow1.calculate_total_error([[trainIN, trainOUT]]), 9))






# train_DF.iloc[0][-10:]      # row 0, last 10 cols - TARGETS
# train_DF.iloc[0][0]         # row 0, col 0
# train_DF.iloc[0]            # entire row 0
# train_DF.iloc[0][1:-10]     # row 0, cols 1 through 10 from last - INPUTS






