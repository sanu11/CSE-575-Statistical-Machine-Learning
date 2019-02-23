import numpy as np
import csv
import math
import random
import matplotlib.pyplot as plt


def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def loss( h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def funMean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stddev(numbers):
	avg = funMean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def calGaussianParameters(X):
	parameters = [(funMean(attribute), stddev(attribute)) for attribute in zip(*X)]
	return parameters


def separateClass(X,Y):
	separated = {0:[],1:[]}
	rows = len(X)
	for i in range(0,rows):
		separated[Y[i][0]].append(X[i])

	return separated


def calculatePdf(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculatePdf(x, mean, stdev)
	return probabilities


def naiveBayes(X,Y):
	data = separateClass(X,Y)
	classWiseParameters = {}
	for classValue, classWiserows in data.iteritems():
		classWiseParameters[classValue] = calGaussianParameters(classWiserows)

	return classWiseParameters

	
def predictNaiveBayes(classWiseParameters,X,Y):
	predictions = []
	for row in X:
		probabilities  = calculateClassProbabilities(classWiseParameters,row)
		if(probabilities[0]>probabilities[1]):
			predictions.append(0)
		else:
			predictions.append(1)
	return predictions
	

def calculateAccuracy(Y,predictions):
	correct = 0

	for x in range(len(Y)):
		if Y[x] == predictions[x]:
			correct += 1
	return (correct/float(len(Y))) * 100.0

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
		# if(i%1000==0):
		# 	print loss(h,y)

	return w


def predictLR(w,X,Y):
	z = np.dot(X,w)
	h=sigmoid(z)
	# print len(h)
	predictions = []
	for value in h:
		if value >=0.5:
			predictions.append(1)
		else:
			predictions.append(0)
	return predictions
	
def LRandNB(X,Y,testX,testY):

	Y = Y.reshape(Y.shape[0],1)

	# --------------------Logistic Regression--------------------
	finalWeights = LogisticRegression(X,Y,lr,iterations)
	predictions = predictLR(finalWeights,testX,testY)
	accuracyLR =  calculateAccuracy(predictions,testY)
	print "Accuracy for LR is ", accuracyLR

	# --------------------Naive Bayes--------------------
	classWiseParameters = naiveBayes(X[:,1:],Y)
	predictions = predictNaiveBayes(classWiseParameters,testX[:,1:],testY)
	accuracyNB = calculateAccuracy(testY,predictions)
	print "Accuracy  for naiveBayes is ", accuracyNB


def readData(fileName):
	X = []
	y = []
	with open(fileName, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			row = [float(num) for num in row]
			X.append(row)

	mainData = list(X)
	# shuffle data
	X = np.array(X)
	XList,YList = splitData3Fold(X)
	XMain =[]
	# add 0th column as 1 to X

	for X in XList:
		X0 = np.zeros((X.shape[0],1))
		XMain.append(np.concatenate((X0,X),axis=1))
	return mainData,XMain,YList


def generateSamples(class1):
	x1 = np.random.normal(class1[0][0], class1[0][1], 400)
	x2 = np.random.normal(class1[1][0], class1[1][1], 400)
	x3 = np.random.normal(class1[2][0], class1[2][1], 4000)
	x4 = np.random.normal(class1[3][0], class1[3][1], 400)
	x = [x1,x2,x3,x4]
	mean1 = np.mean(x1)
	mean2 = np.mean(x2)
	mean3 = np.mean(x3)
	mean4 = np.mean(x4)
	mean = [mean1,mean2,mean3,mean4]
	std1 = np.std(x1)
	std2 = np.std(x2)
	std3 = np.std(x3)
	std4 = np.std(x4)
	std = [std1,std2,std3,std4]

	return x,mean,std


def splitData3Fold(X):

	np.random.shuffle(X)

	# split x and y
	X = np.array(X)
	temp = X[:,0:4]
	y = X[:,4:]
	X = temp

	# three fold validation - split data 	
	rows = X.shape[0]
	part = rows/3

	#split into three parts
	X1 = X[0:part,:]
	X2 = X[part:2*part,:]
	X3 = X[part*2:]

	y1 = y[0:part]
	y2 = y[part:part*2]
	y3 = y[part*2:]

	# print X1.shape,X2.shape,X3.shape
	
	XList = [X1,X2,X3]
	YList = [y1,y2,y3]

	return XList,YList

if __name__ == "__main__":
	iterations = 10000
	lr=0.01
	mainData,X, Y = readData('dataset.csv')
	# input data
	X1 = X[0]
	X2 = X[1]
	X3 = X[2]

	# labels
	y1 = Y[0]
	y2 = Y[1]
	y3 = Y[2]
	# print Y

#  ------------------3 fold LR and NB--------------------------
	#  using last group as test data
	print "-----------------Naive Bayes and Logistic Regression  algorithms using 3 fold cross validation-----------------"
	print "Using 1st group for test and other two for train"
	X = np.concatenate((X2,X3),axis=0)
	Y = np.concatenate((y2,y3),axis =0)
	LRandNB(X,Y,X1,y1)

	# using 2nd group as test data
	print "\nUsing 2nd group for test and other two for train"
	X = np.concatenate((X1,X3),axis=0)
	Y = np.concatenate((y1,y3),axis =0)
	LRandNB(X,Y,X2,y2)

	# using 3rd group as test data
	print "\nUsing 3rd group for test and other two for train"
	X = np.concatenate((X1,X2),axis=0)
	Y = np.concatenate((y1,y2),axis =0)
	LRandNB(X,Y,X3,y3)


# ---------------- 2nd Que ------------------------------
	print ""
	print "-----------------Q2.Accuracies for different fractions-----------------\n"
	mainTrain,test = splitDataset(mainData,0.7)
	# print test
	test =  np.array(test)
	test = np.concatenate((np.zeros((test.shape[0],1)),test),axis=1)
	fractionAccuracyLR = []
	fractionAccuracyNB = []
	fractions = [.01, .02, .05, .1, .625, 1.0]
	xlabels = [ (fraction) for fraction in fractions ]
 	for fraction in fractions:
		
		print "For fraction ", fraction
		# --------------------Logistic Regression--------------------
		accuracyLR = []
		accuracyNB =[] 
		for i in range(0,5):
			train,copy = splitDataset(mainTrain,fraction)
			train = np.array(train)
			train = np.concatenate((np.zeros((train.shape[0],1)),train),axis=1)

			finalWeights = LogisticRegression(train[:,:5],train[:,5:],lr,iterations)
			predictions = predictLR(finalWeights,test[:,:5],test[:,5:])
			# print predictions,test[:,5:],"hi"
			accuracy =  calculateAccuracy(predictions,test[:,5:])
			accuracyLR.append(accuracy)
			# print "Accuracy for LR is ", accuracyLR

			# --------------------Naive Bayes--------------------
			classWiseParameters = naiveBayes(train[:,1:5],train[:,5:])
			predictions = predictNaiveBayes(classWiseParameters,test[:,1:5],test[:,5:])
			accuracy = calculateAccuracy(test[:,5:],predictions)
			accuracyNB.append(accuracy)
			# print "Accuracy  for naiveBayes is ", accuracyNB

		averageAccuarcyLR = sum(accuracyLR)/5
		averageAccuarcyNB = sum(accuracyNB)/5
		print "Logistic Regression accuracy ", averageAccuarcyLR
		print "Naive Bayes accuracy " , averageAccuarcyNB
		fractionAccuracyLR.append(averageAccuarcyLR)
		fractionAccuracyNB.append(averageAccuarcyNB)
	# print fractionAccuracyNB,fractionAccuracyLR
	# fig, axs = plt.plot()
	# fig.suptitle("\nTraining size vs Accuracy\n")
	ticks = [i for i in range(0, len(xlabels))]
	# print ticks,xlabels

	plt.xlabel("Fractions")
	plt.ylabel("Accuracy")
	plt.plot(fractions,fractionAccuracyLR,'r',label="Logistic Regression")
	plt.plot(fractions,fractionAccuracyNB,'b',label="naive Bayes")
	# plt.xscale('log')
	locs, labels = plt.xticks()
	# print locs,"loc",labels,"labels"
	plt.xticks(xlabels,rotation=90)
	# plt.set_xlabel('fraction of Training data')
	# plt.set_ylabel('Accuracy')
	plt.legend()
	plt.show()
	locs, labels = plt.xticks()
	# print locs,labels

	# -------------- Question 3rd----------------------
	print ""
	print "-----------------Q3.Naive Bayes as Generative model----------------- "
	Xlist,Ylist = splitData3Fold(mainTrain)
	X1= Xlist[0] + Xlist[1]
	X2  = Xlist[1] + Xlist[2]
	X3 = Xlist[0] + Xlist[2] 

	Y1 = Ylist[0]
	Y2 = Ylist[1]
	Y3 = Ylist[2]
	l1 = [X1,X2,X3]
	l2 = [Y1,Y2,Y3]
	i =1
	for (X,Y) in zip(l1,l2):
		print "\nFOLD: ",i
		X = np.array(X)
		Y = np.array(Y)
		classWiseParameters = naiveBayes(X,Y)
		class1 = classWiseParameters[1]
		# print class1
		x,mean,std = generateSamples(class1)

		print "Mean:"
		print "Feature 1:"
		print "Random generated samples: " ,mean[0]
		print "Trained model mean: ", class1[0][0]

		print "Feature 2:"
		print "Random generated samples: " ,mean[1]
		print "Trained model mean: ", class1[1][0]
		
		print "Feature 3:"
		print "Random generated samples: " ,mean[2]
		print "Trained model mean: ", class1[2][0]
		
		print "Feature 4:"
		print "Random generated samples: " ,mean[3]
		print "Trained model mean: ", class1[3][0]

		print "\nStandard Deviation:"
		print "Feature 1:"
		print "Random generated samples: " ,std[0]
		print "Trained model mean: ", class1[0][1]

		print "Feature 2:"
		print "Random generated samples: " ,std[1]
		print "Trained model mean: ", class1[1][1]
		
		print "Feature 3:"
		print "Random generated samples: " ,std[2]
		print "Trained model mean: ", class1[2][1]
		
		print "Feature 4:"
		print "Random generated samples: " ,std[3]
		print "Trained model mean: ", class1[3][1]
		i+=1
		
		
