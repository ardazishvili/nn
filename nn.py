import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]]).T
y = np.array([[0,1,1,0]])

def gradient_descent(epochs, rate):
    weights_1 = 2 * np.random.randn(4,3) - 1
    weights_2 = 2 * np.random.randn(1,4) - 1
    for i in range(epochs):
        z_1 = weights_1.dot(X)
        a_1 = sigmoid(z_1)
        #print('a_1.shape - {0}'.format(a_1.shape))
        z_2 = weights_2.dot(a_1)
        a_2 = sigmoid(z_2)
        #print('a_2.shape - {0}'.format(a_2.shape))
        
        delta_2 = (a_2 - y) * sigmoid_prime(z_2)
        #print('delta_2.shape - {0}'.format(delta_2.shape))
        delta_1 = (weights_2.T.dot(delta_2)) * sigmoid_prime(z_1)
        #print('delta_1.shape - {0}'.format(delta_1.shape))
        
        weights_2 = weights_2 - rate * delta_2.dot(a_1.T)
        weights_1 = weights_1 - rate * delta_1.dot(X.T)
        
        if (i % (epochs / 10) == 0):
            print ("Error:" + str(np.mean(np.abs(y - a_2))))
        
gradient_descent(100000, 1)

