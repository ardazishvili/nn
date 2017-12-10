import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

X = np.array([[0,0,1,1],
              [0,1,0,1],
              [1,1,1,1]])
y = np.array([[0,1,1,0]])

def forward_prop(X, weights):
    z = []
    a = []
    
    z.append(weights[0].dot(X))
    a.append(sigmoid(z[0]))
    
    for i in range(1, len(weights)):
        z.append(weights[i].dot(a[i-1]))
        a.append(sigmoid(z[i]))
    
    return z, a

def gradient_descent(epochs, rate):
    weights_1 = 2 * np.random.randn(4,3) - 1
    weights_2 = 2 * np.random.randn(1,4) - 1
    for i in range(epochs):
        z, a = forward_prop(X, [weights_1, weights_2])
        
        delta_2 = (a[1] - y) * sigmoid_prime(z[1])
        delta_1 = (weights_2.T.dot(delta_2)) * sigmoid_prime(z[0])
        
        weights_2 = weights_2 - rate * delta_2.dot(a[0].T)
        weights_1 = weights_1 - rate * delta_1.dot(X.T)
        
        if (i % (epochs / 10) == 0):
            print ("Error:" + str(np.mean(np.abs(y - a[1]))))
        
gradient_descent(100000, 1)

