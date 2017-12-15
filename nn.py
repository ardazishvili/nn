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

#X = np.array([[0,0,1,1],
              #[0,1,0,1],
              #[1,1,1,1]])
#y = np.array([[0,1,1,0]])

mnist = fetch_mldata('MNIST original', data_home='/home/roman/experiments/nn/')
q = 20
X = mnist.data.T[:, :q] / np.max(mnist.data)
y = mnist.target[np.newaxis,:][:, :q]

            
class Network:
    def __init__(self, layers, activation=Sigmoid()):
        self.biases = [np.random.randn(m, 1) for m in layers[1:]]
        self.weights = [np.random.randn(n, m) for n,m in zip(layers[1:], layers[:-1])]
        self.a = activation
        
        
    def forward_prop(self, X):
        
        #print('\n')
        z = []
        a = [X]
        
        
        for i, [w, b] in enumerate(zip(self.weights, self.biases)):
            #print(w.shape)
            #print(b.shape)
            #print(a[i].shape)
            z.append(w.dot(a[i]) + b)
            a.append(self.a.f(z[i]))
        #print('\n')
        return z, a[1:]


    def back_prop(self, X, y, z, a):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        a = [X] + a
        
        delta = (a[-1] - y) * self.a.prime(z[-1])
        nabla_b[-1] = delta
        print(a[-2].shape)
        nabla_w[-1] = delta.dot(a[-2].T)
        
        for layer in range(2, len(self.weights) + 1):
            print('sdf')
            z = z[-layer]
            sp = self.a.prime(z)
            delta = self.weights[-layer+1].T.dot(delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = delta.dot(a[-layer-1].T)
        return nabla_b, nabla_w
        
        #for i in range(1, len(self.weights)):
            #delta.append((self.weights[-i].T.dot(delta[i-1])) * self.a.prime(z[-1-i]))
        #return list(reversed(delta))


    def update_weights(self, X, nabla_b, nabla_w, rate):
        #nabla_b = [np.sum(nb, axis=1)[:,np.newaxis] for nb in nabla_b]
        #nabla_w = [np.sum(nb, axis=1)[:,np.newaxis] for nb in nabla_b]
        
        
        self.weights = [w - (rate/len(X)) * nw for w, nw in zip(self.weights, nabla_w)]
        
        print(nabla_w[0].shape)
        print(nabla_w[1].shape)
        #print(self.biases[0].shape)
        #print(self.biases[1].shape)
        self.biases = [b - (rate/len(X)) * nb for b, nb in zip(self.biases, nabla_b)]
        #print(self.biases[0].shape)
        #print(self.biases[1].shape)
        print('\n')
   
   
    def gradient_descent(self, epochs, rate, X, y, batch_size):
        cost = []
        for i in range(epochs):
            for j in range(int(X.shape[1] / batch_size)):
                X_batch = X[:, j*batch_size:(j+1)*batch_size]
                y_batch = y[:, j*batch_size:(j+1)*batch_size]
                z, a = self.forward_prop(X_batch)
                nabla_b, nabla_w = self.back_prop(X_batch, y_batch, z, a)
                self.update_weights(X_batch, nabla_b, nabla_w, rate)

            z, a = self.forward_prop(X)
            #cost.append(cost_func(y, a[-1]))
            #print(cost_func(y, a[-1]))
        return cost
                
epochs = 1
batch_size = 10
n = Network([784, 30, 1])

x = np.arange(epochs)
y = n.gradient_descent(epochs, 1, X, y, batch_size)


#fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(x, y)
#plt.ylim(0, y[1])
#plt.show()

