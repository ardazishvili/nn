import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

X = np.array([[0,0,1,1],
              [0,1,0,1],
              [1,1,1,1]])
y = np.array([[0,1,1,0]])


def forward_prop(X, weights, biases):
    z = []
    a = []
    
    z.append(weights[0].dot(X) + biases[0])
    a.append(sigmoid(z[0]))
    
    for i in range(1, len(weights)):
        z.append(weights[i].dot(a[i-1]) + biases[i])
        a.append(sigmoid(z[i]))
    
    return z, a


def back_prop(y, z, a, weights):
    delta = []
    delta.append((a[-1] - y) * sigmoid_prime(z[-1]))
    
    for i in range(1, len(weights)):
        delta.append((weights[-i].T.dot(delta[i-1])) * sigmoid_prime(z[-1-i]))
    
    return list(reversed(delta))


def update_weights(X, weights, biases, delta, a, rate):
    w = []
    b = []
    
    w.append(weights[0] - rate * delta[0].dot(X.T))
    b.append(biases[0] - rate * np.sum(delta[0], axis=1))
    for i in range(1, len(weights)):
        w.append(weights[i] - rate * delta[i].dot(a[i-1].T))
        b.append(biases[i]  - rate * np.sum(delta[i], axis=1))
        
    return w, b
    

def gradient_descent(epochs, rate):
    weights_1 = 2 * np.random.randn(4, 3) - 1
    biases_1 = 2 * np.random.randn(1, 4) - 1
    
    weights_2 = 2 * np.random.randn(1, 4) - 1
    biases_2 = 2 * np.random.randn(1, 1) - 1
    for i in range(epochs):
        z, a = forward_prop(X, [weights_1, weights_2], [biases_1, biases_2])
        delta = back_prop(y, z, a, [weights_1, weights_2])
        [weights_1, weights_2], [biases_1, biases_2] = update_weights(X, [weights_1, weights_2], [biases_1, biases_2], delta, a, rate)
        
        
        if (i % (epochs / 10) == 0):
            print ("Error:" + str(np.mean(np.abs(y - a[1]))))
        
gradient_descent(50000, 1)

