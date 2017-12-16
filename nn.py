import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


class Sigmoid:
    def f(self, z):
        z = 1.0 / (1 + np.exp(-z))
        return z

    def prime(self, z):
        return self.f(z) * (1 - self.f(z))

    
def cost_func(a, b):
    return np.sum(np.power(a - b, 2)) / (2 * len(a))

mnist = fetch_mldata('MNIST original', data_home='/home/roman/experiments/nn/')
q = 1000
X = mnist.data.T[:, :q] / np.max(mnist.data)
y = mnist.target[np.newaxis,:][:, :q]

X_test = mnist.data.T[:, 60000:] / np.max(mnist.data)
y_test = mnist.target[np.newaxis,:][:, 60000:]

            
class Network:
    def __init__(self, layers, activation=Sigmoid()):
        self.biases = [np.random.randn(m, 1) for m in layers[1:]]
        self.weights = [np.random.randn(n, m) for n,m in zip(layers[1:], layers[:-1])]
        self.a = activation
        
        
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
        
        z, a = self.forward_prop(X)
        
        delta = (a[-1] - y) * self.a.prime(z[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(a[-2].T)
        
        for layer in range(2, len(self.weights) + 1):
            z = z[-layer]
            sp = self.a.prime(z)
            delta = self.weights[-layer+1].T.dot(delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = delta.dot(a[-layer-1].T)
        return nabla_b, nabla_w


    def update_weights(self, X_batch, y_batch, rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for i in range(X_batch.shape[1]):
            X = X_batch[:, i][:,np.newaxis]
            y = y_batch[:, i][:,np.newaxis]
            delta_nabla_b, delta_nabla_w = self.back_prop(X, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w - (rate/len(X_batch))*nw for w, nw in zip(self.weights, nabla_w)]
            
        self.biases = [b - (rate/len(X_batch))*nb for b, nb in zip(self.biases, nabla_b)]
   
   
    def gradient_descent(self, epochs, rate, X, y, batch_size, Xt, yt):
        cost = []
        for i in range(epochs):
            for j in range(int(X.shape[1] / batch_size)):
                X_batch = X[:, j*batch_size:(j+1)*batch_size]
                y_batch = y[:, j*batch_size:(j+1)*batch_size]
                self.update_weights(X_batch, y_batch, rate)


            self.check_results(Xt, yt)
            z, a = self.forward_prop(X)
            c = cost_func(y, a[-1])
            cost.append(c)
            print('cost = ' + str(c) + '\n')
            
        return cost
    
    
    def check_results(self, X_test, y_test):
        counter = 0
        for i in range(X_test.shape[1]):
            X = X_test[:, i][:,np.newaxis]
            y = y_test[:, i][:,np.newaxis]
            z, a = self.forward_prop(X)
            if (np.argmax(a[-1]) == int(y)):
                counter +=1
        print(counter)
                
epochs = 30
batch_size = 1000
n = Network([784, 30, 10])

x = np.arange(epochs)
y = n.gradient_descent(epochs, 3.0, X, y, batch_size, X_test, y_test)


#fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(x, y)
#plt.ylim(0, y[1])
#plt.show()

