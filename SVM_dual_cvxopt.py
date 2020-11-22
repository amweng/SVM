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

print(X.shape)
print(y.shape)
plt.show()

def kernel_svm(X, y): 
    
    y_matrix = y[:,np.newaxis]
    kernel = X.dot(X.T)

    y_yTranspose = y_matrix.dot(y_matrix.T)

    P_numpy = y_yTranspose * kernel

    # get our Q to be in the form of the solver
    q_numpy = -1 * np.ones((len(X),1))
    G_numpy = -1 * np.identity(len(X))
    h_numpy = np.zeros(len(X))
    A_numpy = y_matrix.T
    b_numpy = np.zeros(1)

    # convert them to the required matrix form
    q = cvxopt.matrix(q_numpy)
    P = cvxopt.matrix(P_numpy)
    h = cvxopt.matrix(h_numpy)
    A = cvxopt.matrix(A_numpy)
    b = cvxopt.matrix(b_numpy)
    # we need to flip the inequality and use the identity matrix - per mit notes
    G = cvxopt.matrix(G_numpy)

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas = np.where(alphas < 0.001, 0, alphas)


    return alphas

# fit svm dual classifier
alphas = kernel_svm(X, y)


def compute_classification_boundary (X, y, alpha):
    

    # compute our kernel values
    kernel = X.dot(X.T)


    alphas = np.array(alpha)

    #multiply alphas times y elementwise
    alpha_y = np.multiply(alphas,y[:,np.newaxis])

    # and same with ws
    ws = np.multiply(alpha_y,X)

    #sum them to get w
    w = np.sum(ws,0)

    # collect our support points
    support_points = []
    support_ys = []
    support_alphas = []
    
    # alpha > 0 = on the margin
    for i in range(len(alpha)):
        if alpha[i] > 0:
            support_point = X[i,:]
            support_points.append(support_point)
            support_ys.append(y[i])
            support_alphas.append(np.array(alpha[i]))

    support_x = np.array(support_points)
    support_y = np.array(support_ys)
    support_alpha = np.array(support_alphas)

    # calculate w0
    support_kernel = support_x.dot(support_x[0].T)
    support_alpha_y = np.multiply(support_alpha,support_y[:,np.newaxis])
    support_to_be_summed = np.multiply(support_alpha_y,support_kernel[:,np.newaxis])

    w0 = support_y[0] - np.sum(support_to_be_summed,0)

    
    return w, w0



w, w0 = compute_classification_boundary(X, y, alphas)

print(w)
print(w0)

def print_support_vectors(X,y,alphas,w,w0):
    wdual = w
    wdual_T = wdual.T
    w0dual = w0
    
    # collect our support points
    support_points = []
    support_ys = []
    support_alphas = []
    
    for i in range(len(alphas)):
        if alphas[i] > 0:
            support_point = X[i,:]
            support_points.append(support_point)
            support_ys.append(y[i])
            support_alphas.append(np.array(alphas[i]))


    support_margins_arr = []

    # for each support point, calculate the functional margin
    for i in range(len(support_points)):
        val = float(support_ys[i] * (wdual_T.dot(support_points[i]) + w0dual))
        support_margins_arr.append(val)

    support_examples = np.array(support_points)

    support_margins = np.array(support_margins_arr)
    np.set_printoptions(precision=20)
    print("support Ys: " + str(support_ys))
    print("points alpha > 0 : " + str(support_examples))
    print("functional margins: " + str(support_margins))
    print("support alphas: " + str(np.array(support_alphas)))

def K(xi, xj):
    return np.dot(xi,xj)

def g_dual(x):

    # use our hypothesis values to calculate prediction
    w0 = 0.46118097
    support_alphas = np.array([1.02105592,1.25118428,2.2722402])
    support_ys = np.array([1.0, 1.0, -1.0])
    support_xs = np.array([[2.11457352, 1.5537852 ],[2.51879639, 1.91565724],[1.71138733, 2.45204836]])

    # generate summation
    support_alphas_ys = np.multiply(support_alphas,support_ys)
    hypothesis_kernel = K(support_xs,x)

    g_to_be_summed = np.multiply(support_alphas_ys,hypothesis_kernel)

    # sum and add w0
    gdual = np.sum(g_to_be_summed,0) + w0

    
    return gdual

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
    

examples = [[3.0,1.5],[1.2,3.0]]



for example in examples:
    if g_dual(example) > 0 :
        print("example: " + str(example) + " predicted to be class 1.")
    else:
        print("example: " + str(example) + " predicted to be class -1.")



