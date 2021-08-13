# https://github.com/Fabbricatore/Replicated-SGD/blob/master/index.md

"""
Created on Wed Jun 19 11:19:10 2019

@author: Fabbricatore
"""

# No fancy libraries

import numpy as np
import matplotlib.pyplot as plt

zz = np.load("x_train.npy")
tt = np.load("t_train.npy")

# N     = size of Training Set
# B     = batch size
# D_in  = dimension of input vectors
# H     = neurons in Hidden Layer
# D_out = dimension of output vectors
N, B, D_0, D_in, H, D_out = 60000, 50, 784, 100, 30, 10

# Replica number
R = 7

# List for the Network Replica
nets = list()

# Learning rates, the second one in for the replica coupling
learning_rate = 1e-7
learning_rate2 = 1e-3

# Number of learnng epochs
times = 100

# Initializing plots
a = np.zeros(times)
b = np.zeros(times)
c = np.zeros(times)
ar = np.zeros(times)
br = np.zeros(times)
d = np.zeros(times)
dr = np.zeros(times)

# Initializing random training set
xx = np.load("x_train.npy") / 255
yy = np.random.randn(N, D_out)

for i in range(N):
    yy[i] = (np.arange(10) == tt[i]).astype(np.int) * 1000


# NETWORK class
class net:

    def __init__(self, D_0, D_in, H, D_out):
        self.D_0 = D_0
        self.D_in = D_in
        self.D_out = D_out
        self.H = H
        self.w0 = np.random.randn(self.D_0, self.D_in)
        self.w1 = np.random.randn(self.D_in, self.H)
        self.w2 = np.random.randn(self.H, self.D_out)

    def work(self, x, y):
        # Forward
        h0 = x.dot(self.w0)
        h0_relu = np.maximum(h0, 0)
        h = h0_relu.dot(self.w1)
        h_relu = np.maximum(h, 0)
        self.y_pred = h_relu.dot(self.w2)

        # Bacward
        grad_y_pred = 2.0 * (self.y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(self.w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = h0_relu.T.dot(grad_h)
        grad_h0_relu = grad_h.dot(self.w1.T)
        grad_h0 = grad_h0_relu.copy()
        grad_h0[h0 < 0] = 0
        grad_w0 = x.T.dot(grad_h0)

        # Update weights
        self.w0 -= learning_rate * grad_w0
        self.w1 -= learning_rate * grad_w1
        self.w2 -= learning_rate * grad_w2

        self.gr0 = abs(np.mean(learning_rate * grad_w0))
        self.gr1 = abs(np.mean(learning_rate * grad_w1))
        self.gr2 = abs(np.mean(learning_rate * grad_w2))

        return self.y_pred

    def grR(self):
        # Replicas update
        self.w0 -= learning_rate2 * (gradR()[0] - self.w0)
        self.w1 -= learning_rate2 * (gradR()[1] - self.w1)
        self.w2 -= learning_rate2 * (gradR()[2] - self.w2)

        self.gr0R = abs(np.mean(learning_rate2 * gradR()[0]))
        self.gr1R = abs(np.mean(learning_rate2 * gradR()[1]))
        self.gr2R = abs(np.mean(learning_rate2 * gradR()[2]))
        # print(self.gr1, self.gr1R)


# Create R Replicas
for i in range(R):
    nets.append(net(D_0, D_in, H, D_out))


# Gradient for the Replica coupling
def gradR():
    gr0, gr1, gr2 = np.zeros([D_0, D_in]), np.zeros([D_in, H]), np.zeros([H, D_out])
    for i in range(R):
        gr0 += nets[i].w0 / R
        gr1 += nets[i].w1 / R
        gr2 += nets[i].w2 / R
    return gr0, gr1, gr2


def loss(y_pred, y):
    return np.square(y_pred - y).sum()


############################   Let's test it   ########################################


for k in range(1):  # Repeat the test 100 times

    for i in range(R):
        nets[i].w1 = np.random.randn(D_in, H)  # Randomly initialize weights every time
        nets[i].w2 = np.random.randn(H, D_out)
        nets[i].w0 = np.random.randn(D_0, D_in)

    for j in range(times):

        r = np.random.randint(N, size=B)  # Randomly pick from Data set
        x = xx[r]
        y = yy[r]

        for i in range(R):  # let all replicas work
            nets[i].work(x, y)

        for i in range(R):
            nets[i].grR()  # Update weights with new rule

        a[j] += nets[0].gr1  # Save averaged results
        b[j] += nets[0].gr2
        d[j] += nets[0].gr0
        ar[j] += nets[0].gr1R
        br[j] += nets[0].gr2R
        dr[j] += nets[0].gr0R
        c[j] += loss(y, nets[0].y_pred)

plt.plot(a * 50, 'r')
plt.plot(b)
plt.plot(c / 100000000)
plt.ylabel('rearning rate')
plt.xlabel(' iterations ')
plt.show()

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(3, 2, 1).plot(b)
plt.title('w3')
fig.add_subplot(3, 2, 3).plot(a, 'r')
plt.title('w2')
fig.add_subplot(3, 2, 5).plot(d, 'g')
plt.title('w3')
fig.add_subplot(3, 2, 2).plot(br)
plt.title('w1 R')
fig.add_subplot(3, 2, 4).plot(ar, 'r')
plt.title('w2 R')
fig.add_subplot(3, 2, 6).plot(dr, 'g')
plt.title('w3 R')

# Printing numbers
fag = plt.figure(figsize=(15, 3))

for i in range(10):
    axs = fag.add_subplot(1, 10, i + 1)  # (row, col, number inside (1<x<row*col))
    axs.imshow(np.reshape(x[i, :], (28, 28)), cmap="gray")
    axs.set_xlabel(np.argmax(nets[0].y_pred[i]))
    axs.set_title(np.argmax(y[i]))
    axs.set_xticks([]);
    axs.set_yticks([])