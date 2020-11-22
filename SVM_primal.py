import numpy as np
import csv
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys
import cvxopt
import matplotlib.pyplot as plt


X = genfromtxt('X.csv', delimiter=',') # reading in the data matrix
y = genfromtxt('y.csv', delimiter=',') # reading in the labels
idx_1 = np.where(y == 1)
idx_0 = np.where(y == -1)

plt.figure(figsize=(15,9))
plt.scatter(X[idx_1,0], X[idx_1,1], s=30, c='b', marker="o")
plt.scatter(X[idx_0,0], X[idx_0,1], s=30, c='r', marker="o")
plt.xlabel('x1')
plt.ylabel('x2');


plt.plot
plt.show()

from cvxopt import matrix, solvers

def linear_svm(X,y):
    solvers.options['show_progress'] = False

    # generate the identity matrix
    y = y.reshape(len(y),1)
    I = np.identity(len(X[0])+1)
    I[0,0] = 0

    oneColumn = np.ones(len(X)).reshape(len(X),1)
    ones_and_x = np.concatenate((oneColumn,X),1)

    # prepare the values for insertion into cvxopt solver
    G_numpy = - np.multiply(y,ones_and_x)
    h_numpy = -1 * np.ones(len(y)).reshape(len(y),1)
    P_numpy = np.diag([0,1,1])
    
    # convert to cvxopt matrix
    G = matrix(G_numpy)
    P = matrix(P_numpy, tc = 'd') 
    q = matrix(np.zeros(len(X[0])+1))
    h = matrix(h_numpy)

    # solve
    sol = cvxopt.solvers.qp(P, q, G, h)

    weights = sol['x']
    return weights

# # fit svm classifier
weights = linear_svm(X, y)
w0 = weights[0]
w = weights[1:3]

print("w0 = " + str(w0))
print("w1,w2 = " + str(w[0]) + " , " + str(w[1]))


def plot_data_with_classification_boundary(X, y, w, w0, fig_size=(15, 9), labels=['x1', 'x2']):
    COLORS = ['blue', 'red']
    unique = np.unique(y)
    


    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == -1)

    plt.figure(figsize=(15,9))
    plt.scatter(X[idx_1,0], X[idx_1,1], s=30, c='b', marker="o")
    plt.scatter(X[idx_0,0], X[idx_0,1], s=10, c='r', marker="o")
    plt.xlabel('x1')
    plt.ylabel('x2');
        
    ## Write code here ##
            
    slope = -w[0] / w[1]
    intercept = -w0 / w[1]
    x = np.arange(0, 6)
    plt.plot(x, x * slope + intercept, 'k-')
    plt.grid()
    plt.show()

# plotting the points and decision boundary   
plot_data_with_classification_boundary(X, y, w, w0)


def calculate_support_vectors(X,y,w,w0):

    w = np.array(w)
    wTranspose = w.T
    fmargins = []

    for i in range(len(X)):
        xi = X[i,:].reshape(len(X[0]),1)
        d = y[i]*(wTranspose.dot(xi) + w0)[0]
        fmargin = d[0]
        # print(fmargin)
        example = (X[i,0],X[i,1])
        fmargins.append([fmargin,example])

        

    sortedMargins = sorted( fmargins , key=lambda x: x[0])
    print("margin, points: " + str(sortedMargins[:5]))
    return 


def g_primal(x):
    
    # use the generated hypothesis to predict the examples
    w = np.array([1.42,-1.59])
    w0 = 0.46118098859684925
    wTranspose = w.T
    xi = x.reshape(len(X[0]),1)
    gamma = (wTranspose.dot(xi) + w0)

    # classify
    if gamma > 0:
        return 1
    if gamma < 0:
        return -1
    
    return 0


Xclassify = np.array([[3.0,1.5],[1.2,3.0]])

for example in Xclassify:
    print("example " + str(example) + " predicted class: " + str(g_primal(example)))

