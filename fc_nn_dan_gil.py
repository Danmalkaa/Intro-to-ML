
import numpy as np
import matplotlib.pyplot as plt
import math


def sig(x):
  return 1 / (1 + np.exp(-x))

def sig_der(x):
    return sig(x)*(1-sig(x))

# part a
error_arr = []
eta=2
X=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
           [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X=np.transpose(X)
T=np.array([0, 1, 1, 1, 0, 0, 0, 1])
X=np.vstack((np.ones(8), X))#now X first row is ones#
for j in range(100):
    W_first=np.random.normal(size=12)
    W_mat=W_first.reshape(3,4)
    W_seclayer=np.transpose(np.random.normal(size=4))
    W_vector=np.concatenate((W_first, W_seclayer))

    for i in range(2000):

        W_mat = W_vector[0:12].reshape(3, 4)

        ##calculate the cells values
        a_firstlayer=np.dot(W_mat, X)
        Z=sig(a_firstlayer)
        Z=np.vstack((np.ones(8), Z))#now Z frst row is ones#
        a_seclayer=np.dot(W_vector[12:], Z)
        Y=sig(a_seclayer)

        ##calculate lamda for output layer
        lamda=np.zeros(32).reshape(4, 8)
        lamda[3,:]=sig_der(a_seclayer)*(Y-T)
        ##calculate lamda for second layer
        lamda[0:3,:]=sig_der(a_firstlayer)*np.dot(W_vector[13:].reshape(3,1), lamda[3:,])

        W_grad=np.vstack((X*lamda[0, :], X*lamda[1, :], X*lamda[2, :], Z*lamda[3, :]))
        W_grad=np.sum(W_grad, axis=1)
        W_vector=W_vector-eta*W_grad
    E = (1/8)*np.sum(np.square(Y-T))
    error_arr += [E]

# plot
iterations = np.linspace(0, 100, 100)
plt.figure(1)
plt.plot(iterations, error_arr, 'r', label='3 cells') # plotting t, b separately
plt.legend()
plt.ylabel("MSE")
plt.suptitle("MSE for 3 cells in Hidden layer (iterations) ")
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

# part b
error_arr = []
eta=2
X=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
           [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X=np.transpose(X)
T=np.array([0, 1, 1, 1, 0, 0, 0, 1])
X=np.vstack((np.ones(8), X))#now X first row is ones#
for j in range(100):
    W_first=np.random.normal(size=24)
    W_mat=W_first.reshape(6,4)
    W_seclayer=np.transpose(np.random.normal(size=7))
    W_vector=np.concatenate((W_first, W_seclayer))

    for i in range(2000):

        W_mat = W_vector[0:24].reshape(6, 4)

        ##calculate the cells values
        a_firstlayer=np.dot(W_mat, X)
        Z=sig(a_firstlayer)
        Z=np.vstack((np.ones(8), Z))#now Z frst row is ones#
        a_seclayer=np.dot(W_vector[24:], Z)
        Y=sig(a_seclayer)

        ##calculate lamda for output layer
        lamda=np.zeros(56).reshape(7, 8)
        lamda[6,:]=sig_der(a_seclayer)*(Y-T)
        ##calculate lamda for second layer
        lamda[0:6,:]=sig_der(a_firstlayer)*np.dot(W_vector[25:].reshape(6,1), lamda[6:,])

        W_grad=np.vstack((X*lamda[0, :], X*lamda[1, :], X*lamda[2, :], X*lamda[3, :], X*lamda[4, :], X*lamda[5, :], Z*lamda[6, :]))
        W_grad=np.sum(W_grad, axis=1)
        W_vector=W_vector-eta*W_grad
    E = (1/8)*np.sum(np.square(Y-T))
    error_arr += [E]

# plot
iterations = np.linspace(0, 100, 100)
plt.figure(2)
plt.plot(iterations, error_arr, 'b', label='6 Cells') # plotting t, b separately
plt.legend()
plt.ylabel("MSE")
plt.suptitle("MSE for 6 cells in Hidden layer (iterations) ")
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()
