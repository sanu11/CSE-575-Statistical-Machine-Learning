import numpy as np
import csv
import random

epsilon = 1e-5

def sigmoid(z):
	# print z
	return 1 / (1 + np.exp(-z))

def regresssionLoss( h, y):
    return (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).mean()

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


def LogisticRegression(X,y,lr,iterations):
	# m = w.shape[0]
	w = np.zeros((X.shape[1],1))
	
	for i in range(iterations):
		z = np.dot(X, w)

		h = sigmoid(z)
		temp = np.dot(X.T, (y - h))
		gradient = np.dot(X.T, (y - h)) / y.shape[0]
		n = gradient.shape[0]

		w += lr * gradient
		if(i%1000==0):
			print regresssionLoss(h,y)

	return w


def predictLR(w,X,Y):
	z = np.dot(X,w)
	h=sigmoid(z)
	print "Loss on test" , regresssionLoss(h,Y)
	# print loss

def readData(fileName):
	X = []
	y = []
	with open(fileName, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			row = [float(num) for num in row]
			row.insert(0,0)
			X.append(row[:5])
	return X 

if __name__ == "__main__":
	iterations = 10000
	lr=0.01
	X = readData('dataset.csv')
	print len(X)
	test,train = splitDataset(X,0.3333)
	
	#  using last group as test data
	print "\nUsing 1st group for test and other two for train"
	train = np.array(train)
	test = np.array(test)

	Y1 = train[:,4:]
	X1 = train[:,0:4]

	testX1 = test[:,4:]
	testY1 = test[:,0:4]

	print X1.shape,Y1.shape
	finalWeights = LogisticRegression(X1,Y1,lr,iterations)
	predictLR(finalWeights,testX1,testY1)

	# # using 2nd group as test data
	# print "\nUsing 2nd group for test and other two for train"
	# X = np.concatenate((X1,X3),axis=0)
	# Y = np.concatenate((y1,y3),axis =0)
	# Y = Y.reshape(Y.shape[0],1)
	# print X.shape, Y.shape

	# finalWeights = LogisticRegression(X,Y,lr,iterations)
	# predictLR(finalWeights,X2,y2)

	# # using 3rd group as test data
	# print "\nUsing 3rd group for test and other two for train"
	# X = np.concatenate((X1,X2),axis=0)
	# Y = np.concatenate((y1,y2),axis =0)
	# Y = Y.reshape(Y.shape[0],1)
	# print X.shape, Y.shape

	# finalWeights = LogisticRegression(X,Y,lr,iterations)
	# predictLR(finalWeights,X3,y3)
