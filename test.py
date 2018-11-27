# Python Test client file

import exampleANN as ANN


import pandas as pd
import numpy as np
import time

# read and format input data
# MacBook file location
# train_DF = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/hw2_ANN/MNIST_in_csv/mnist_train.csv')
# iMac file location
train_DF = pd.read_csv('/Users/shimac/Documents/ComputerSci/cs460_ML/hw02/MNIST_in_csv/mnist_train.csv')


print('TRAINING DATA:\n',train_DF.head())

pixels = 28*28
pixArray = []
for x in range(0,pixels + 1):
    if x == 0:
        pixArray.append('target')
    else:
        pixArray.append(x)
train_DF.columns = pixArray
# print('FORMATTED TRAINING DATA:\n',train_DF.head())    

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
print('FORMATTED TRAINING DATA:\n',train_DF.head())    


# epochs
# ep = 500
ep = 5
numInputs = 784					# 785? or 784? <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ERROR index
numOutputs = 10
# numHiddenNeurons = 100
# hiddenWeights = list(np.linspace(-0.01,0.01, (numInputs * numHiddenNeurons)))
# hiddenBias = 1
# outputWeights = list(np.linspace(-0.01,0.01, (numHiddenNeurons * numOutputs)))
# outputBias = 1
# AnnModel = ANN.NeuralNetwork(numInputs, numHiddenNeurons, numOutputs, hiddenWeights, hiddenBias, outputWeights, outputBias)
# pick row numbrer
# rowIndex = 0
# trainOUT = train_DF.iloc[rowIndex][-10:] 		# last 10 columns of targets
# trainIN = train_DF.iloc[rowIndex][1:-10]		# pixel input colummns

# trainIN_list = list(trainIN)
# trainOUT_list = list(trainOUT)

# ************************************************************************** Test Accuracy
# import test data set
test_DF = pd.read_csv('/Users/shimac/Documents/ComputerSci/cs460_ML/hw02/MNIST_in_csv/mnist_test.csv')
print('TEST DATA:\n', test_DF.head())
print('training data length:',len(train_DF.index))
print('test data length:',len(test_DF.index))
print('epocks: ', ep)
print('data rows divided by 9...every 9th row used')
# Test accuracy using varying amounts of hidden nodes
hiddenArray = [1,2,5,10,15,20,40,80]
# hiddenArray = [10]
accuracyArray = []
timeArray = []
for x in hiddenArray:
	start = time.time()	# measure time
	numHiddenNeurons = x
	hiddenWeights = list(np.linspace(-0.01,0.01, (numInputs * numHiddenNeurons)))
	hiddenBias = 1
	outputWeights = list(np.linspace(-0.01,0.01, (numHiddenNeurons * numOutputs)))
	outputBias = 1
	AnnModel = ANN.NeuralNetwork(numInputs, numHiddenNeurons, numOutputs, hiddenWeights, hiddenBias, outputWeights, outputBias)
	print('learning rate:', AnnModel.LEARNING_RATE)
	print('numHiddenNeurons =', numHiddenNeurons)
	# make model for each hiddenArray configuration
	for j in range(ep):
		# print('~~~ training epock: ', j)
		for row in range(0,len(train_DF.index)):
			# data is too big, must slice it down by a fraction
			if row % 9 == 1:
				# print('~~~~~~~~ training row:',row)
				trainINPUT = list(train_DF.iloc[row][1:-10])		# pixel input colummns
				trainOUTPUT = list(train_DF.iloc[row][-10:])
				AnnModel.train(trainINPUT, trainOUTPUT)
	# for this model - calculate accuracy
	correct = 0
	wrong = 0
	for row in range(0,len(test_DF.index)):
		# print('~~~ testing row: ', row)
		# predicted = nnRow1.predict(list(test_DF.iloc[0][1:-10]))  
		predicted = AnnModel.predict(list(test_DF.iloc[row][1:]))			# test data was not reformatted with 10 extra columns
		if predicted == test_DF.iloc[row][0]:
			correct += 1
		else:
			wrong += 1
	percentage = correct / (correct + wrong)
	accuracyArray.append(percentage)
	print('\taccuracy:\t', percentage)
	end = time.time()
	print('\ttime:\t', end - start, ' seconds')
	# print('inspect:\n',AnnModel.inspect())
	timeArray.append(end - start)

for x in range(0, len(hiddenArray)):
	print('hidden nodes: ', hiddenArray[x], '\t accuracy: ', accuracyArray[x], 'time:', timeArray[x])




