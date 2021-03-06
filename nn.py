import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import random


class Sigmoid:
    def f(self, z):
        z = 1.0 / (1 + np.exp(-z))
        return z

    def derivative(self, z):
        return self.f(z) * (1 - self.f(z))


class Quadratic_cost:
    def f(self, a, y):
        return 0.5 * np.mean(np.power(a - y, 2))
    
    def derivative(self, a, y, activation_derivative, arg):
        return (a - y) * activation_derivative(arg)
    
    
class Cross_entropy_cost:
    def f(self, a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y)*np.log(1 - a))) / len(a)
    
    def derivative(self, a, y,  activation_derivative, arg):
        return (a - y)
        

mnist = fetch_mldata('MNIST original', data_home='/home/roman/experiments/nn/')
order = np.arange(70000)
np.random.shuffle(order)

mnist.data = mnist.data.T / np.max(mnist.data)

i = np.argsort(order)
mnist.data = mnist.data[:,i]
mnist.target = mnist.target[i]

all_data = []
for i in range(70000):
    y = np.zeros(10).reshape(10,1)
    yi = mnist.target[i]
    y[int(yi), 0] = 1
    all_data.append((mnist.data[:,i][:,np.newaxis], y))

training_data = all_data[:50000]
test_data = all_data[60000:]



            
class Network:
    def __init__(self, layers, activation=Sigmoid(), cost=Quadratic_cost()):
        self.layers_num = len(layers)
        self.biases = [np.random.randn(m, 1) for m in layers[1:]]
        self.weights = [np.random.randn(n, m) for n,m in zip(layers[1:], layers[:-1])]
        self.a = activation
        self.c = cost
        
        
    def forward_prop(self, X):
        z = []
        a = [X]
        for i, [w, b] in enumerate(zip(self.weights, self.biases)):
            z.append(w.dot(a[i]) + b)
            a.append(self.a.f(z[i]))
        return z, a


    def back_prop(self, X, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        zs, a = self.forward_prop(X)
        
        #delta = (a[-1] - y) * self.a.derivative(zs[-1])
        delta = self.c.derivative(a[-1], y, self.a.derivative, zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(a[-2].T)

        for layer in range(2, self.layers_num):
            z = zs[-layer]
            sp = self.a.derivative(z)
            delta = self.weights[-layer+1].T.dot(delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = delta.dot(a[-layer-1].T)
        return nabla_b, nabla_w


    def update_weights(self, batch, rate):
        n = len(batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for X, y in batch:
            delta_nabla_b, delta_nabla_w = self.back_prop(X, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
            self.weights = [w - (rate/n)*nw for w, nw in list(zip(self.weights, nabla_w))]
            
            self.biases = [b - (rate/n)*nb for b, nb in list(zip(self.biases, nabla_b))]
        
        
    def gradient_descent(self, training_data, epochs, mini_batch_size, rate, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            
        return cost
    
    def evaluate(self, test_data):
        test_results = []
        for x, y in test_data:
            z, am = self.forward_prop(x)
            test_results.append((np.argmax(am[-1]), np.argmax(y)))
            
        counter = 0
        for x, y in test_results:
            if x == y:
                counter += 1
                
        return counter
                
epochs = 30
batch_size = 10
n = Network([784, 30, 10])

y = n.gradient_descent(training_data, epochs, batch_size, 3.0, test_data=test_data)


#fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(x, y)
#plt.ylim(0, y[1])
#plt.show()

