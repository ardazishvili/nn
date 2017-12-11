import numpy as np

class Sigmoid:
    def f(self, z):
        return 1 / (1 + np.exp(-z))

    def prime(self, z):
        return self.f(z) * (1 - self.f(z))

X = np.array([[0,0,1,1],
              [0,1,0,1],
              [1,1,1,1]])
y = np.array([[0,1,1,0]])

            
class Network:
    def __init__(self, layers, activation=Sigmoid()):
        self.biases = [(2 * np.random.randn(m, 1) - 1) for m in layers[1:]]
        self.weights = [(2 * np.random.randn(n, m) - 1) for n,m in zip(layers[1:], layers[:-1])]
        self.a = activation
        
        #print(self.weights[0].shape)
        #print(self.weights[1].shape)
        #print(self.biases[0].shape)
        #print(self.biases[1].shape)
        #print('\n\n')
        
        
    def forward_prop(self, X):
        z = []
        a = []
        z.append(self.weights[0].dot(X) + self.biases[0])
        a.append(self.a.f(z[0]))
        for i in range(1, len(self.weights)):
            z.append(self.weights[i].dot(a[i-1]) + self.biases[i])
            a.append(self.a.f(z[i]))
        return z, a


    def back_prop(self, y, z, a):
        delta = []
        delta.append((a[-1] - y) * self.a.prime(z[-1]))
        for i in range(1, len(self.weights)):
            delta.append((self.weights[-i].T.dot(delta[i-1])) * self.a.prime(z[-1-i]))
        return list(reversed(delta))


    def update_weights(self, X, delta, a, rate):
        w = []
        b = []
        w.append(self.weights[0] - rate * delta[0].dot(X.T))
        b.append(self.biases[0] - rate * np.sum(delta[0], axis=1).reshape( delta[0].shape[0], 1))
        for i in range(1, len(self.weights)):
            w.append(self.weights[i] - rate * delta[i].dot(a[i-1].T))
            b.append(self.biases[i]  - rate * np.sum(delta[i], axis=1))
        self.weights, self.biases =  w, b
   
   
    def gradient_descent(self, epochs, rate, X, y):
        for i in range(epochs):
            z, a = self.forward_prop(X)
            delta = self.back_prop(y, z, a)
            self.update_weights(X, delta, a, rate)

            if (i % (epochs / 10) == 0):
                print ("Error at epoch " + str(i) + " :" + str(np.mean(np.abs(y - a[1]))))
                

n = Network([3, 6 ,1])
n.gradient_descent(50000, 1, X, y)

