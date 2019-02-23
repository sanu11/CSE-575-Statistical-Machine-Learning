import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss( h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
X = [[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]]
X = np.array(X)
y = np.array([1,1,1,0,0,0]).T
m = y.shape[0]
y = y.reshape(m,1)

w1 = np.array([0.0,0.0,0.0,0.0]).T
w2 = np.array([0.0,0.0,1.0,0.0]).T

def LogisticRegression(w,lr,iterations):
	m = w.shape[0]
	w = w.reshape(m,1)
	for i in range(iterations):
		z = np.dot(X, w)

		h = sigmoid(z)
		temp = np.dot(X.T, (y - h))
		gradient = np.dot(X.T, (y - h)) / y.shape[0]
		n = gradient.shape[0]

		w += lr * gradient
		if(i%100000==0):
			print "Loss after, ",i,"th iteration:", loss(h,y)

	return w



if __name__ == "__main__":
	iterations = 1000000
	lr=0.01
	w = w1
	finalWeight = LogisticRegression(w,lr,iterations)
	print "Final weight vector for case (0,0,0,0):\n" , finalWeight
	w = w2
	finalWeight = LogisticRegression(w,lr,iterations)
	print "Final weight vector for case (0,0,1,0):\n" , finalWeight
