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
        
        
    def forward_prop(self, X):
        z = []
        a = [X]
        for i in range(0, len(self.weights)):
            z.append(self.weights[i].dot(a[i]) + self.biases[i])
            a.append(self.a.f(z[i]))
        return z, a[1:]


    def back_prop(self, y, z, a):
        delta = [(a[-1] - y) * self.a.prime(z[-1])]
        for i in range(1, len(self.weights)):
            delta.append((self.weights[-i].T.dot(delta[i-1])) * self.a.prime(z[-1-i]))
        return list(reversed(delta))


    def update_weights(self, X, delta, a, rate):
        a = [X] + a
        for i in range(0, len(self.weights)):
            self.weights[i] -= rate * delta[i].dot(a[i].T)
            self.biases[i] -= rate * np.sum(delta[i], axis=1)[:,np.newaxis]
   
   
    def gradient_descent(self, epochs, rate, X, y):
        for i in range(epochs):
            z, a = self.forward_prop(X)
            delta = self.back_prop(y, z, a)
            self.update_weights(X, delta, a, rate)

            if (i % (epochs / 10) == 0):
                print ("Error at epoch " + str(i) + " :" + str(np.mean(np.abs(y - a[1]))))
                

n = Network([3, 4 ,1])
n.gradient_descent(10000, 1, X, y)

