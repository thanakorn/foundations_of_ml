#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
plt.rcParams.update(plt.rcParamsDefault)

#%% Calculate probability for gaussian distribution
def gauss2D(x, m, C):
    Ci = np.linalg.inv(C)
    dC = np.linalg.det(C)
    num = np.exp(-0.5 * np.dot((x -  m).T, np.dot(Ci, (x-m))))
    den = 2 * np.pi * (dC ** 0.5)
    return num / den
#%% 2D Gaussian Plot
def twoDGaussianPlot(nx, ny, m, C):
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing = 'ij')

    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            Z[i,j] = gauss2D(xvec, m, C)

    return X, Y, Z

#%% Posterior probability plot
def posteriorPlot(nx, ny, m1, C1, m2, C2, P1, P2):
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            num = P1 * gauss2D(xvec, m1, C1)
            den = P1 * gauss2D(xvec, m1, C1) + P2 * gauss2D(xvec, m2, C2)
            Z[i,j] = num /den
    
    return X, Y, Z

#%% 1. Class Boundaries and Posterior Probability
num_data = 200
X = np.random.randn(num_data,2)

y_m1 = [0,3]
y_m2 = [3,2.5]
y_c1 = np.array([[2,1], [1,2]], np.float32)
y_c2 = np.array([[2,1], [1,2]], np.float32)
y_A1 = np.linalg.cholesky(y_c1)
y_A2 = np.linalg.cholesky(y_c2)
y_p1 = 0.5
y_p2 = 0.5

Y1 = (X @ y_A1) + y_m1
Y2 = (X @ y_A2) + y_m2
plt.figure()
plt.scatter(Y1[:,0], Y1[:,1], color = 'r', label = 'C1')
plt.scatter(Y2[:,0], Y2[:,1], color = 'g', label = 'C2')
X1p, Y1p, Z1p = twoDGaussianPlot(100, 100, y_m1, y_c1)
X2p, Y2p, Z2p = twoDGaussianPlot(100, 100, y_m2, y_c2)
plt.contour(X1p, Y1p, Z1p, 3, colors = 'r')
plt.contour(X2p, Y2p, Z2p, 3, colors = 'g')
plt.legend()
plt.title('Y')
plt.savefig('Lab3/y_scatter.png')
plt.show()

plt.figure()
CT_X, CT_Y, CT_Z = posteriorPlot(50, 40, y_m1, y_c1, y_m2, y_c2, y_p1, y_p2)
plt.contour(CT_X, CT_Y, CT_Z, 3)
plt.title('Y Posterior Probability')
plt.savefig('Lab3/y_posterior.png')
plt.show()

z_m1 = [0,3]
z_m2 = [3,2.5]
z_c1 = np.array([[2,1], [1,2]], np.float32)
z_c2 = np.array([[2,1], [1,2]], np.float32)
z_A1 = np.linalg.cholesky(y_c1)
z_A2 = np.linalg.cholesky(y_c2)
z_p1 = 0.7
z_p2 = 0.3

Z1 = (X @ y_A1) + z_m1
Z2 = (X @ y_A2) + z_m2
plt.figure()
plt.scatter(Z1[:,0], Z1[:,1], color = 'r', label = 'C1')
plt.scatter(Z2[:,0], Z2[:,1], color = 'g', label = 'C2')
X1p, Y1p, Z1p = twoDGaussianPlot(100, 100, z_m1, z_c1)
X2p, Y2p, Z2p = twoDGaussianPlot(100, 100, z_m2, z_c2)
plt.contour(X1p, Y1p, Z1p, 3, colors = 'r')
plt.contour(X2p, Y2p, Z2p, 3, colors = 'g')
plt.legend()
plt.title('Z')
plt.savefig('Lab3/z_scatter.png')
plt.show()

plt.figure()
CT_X, CT_Y, CT_Z = posteriorPlot(50, 40, z_m1, z_c1, z_m2, z_c2, z_p1, z_p2)
plt.contour(CT_X, CT_Y, CT_Z, 3)
plt.title('Z Posterior Probability')
plt.savefig('Lab3/z_posterior.png')
plt.show()

w_m1 = [0,3]
w_m2 = [3,2.5]
w_c1 = np.array([[2,0], [0,2]], np.float32)
w_c2 = np.array([[1.5,0], [0,1.5]], np.float32)
w_A1 = np.linalg.cholesky(w_c1)
w_A2 = np.linalg.cholesky(w_c2)
w_p1 = 0.5
w_p2 = 0.5

W1 = (X @ y_A1) + w_m1
W2 = (X @ y_A2) + w_m2
plt.figure()
plt.scatter(W1[:,0], W1[:,1], color = 'r', label = 'C1')
plt.scatter(W2[:,0], W2[:,1], color = 'g', label = 'C2')
X1p, Y1p, Z1p = twoDGaussianPlot(100, 100, w_m1, w_c1)
X2p, Y2p, Z2p = twoDGaussianPlot(100, 100, w_m2, w_c2)
plt.contour(X1p, Y1p, Z1p, 3, colors = 'r')
plt.contour(X2p, Y2p, Z2p, 3, colors = 'g')
plt.legend()
plt.title('W')
plt.savefig('Lab3/w_scatter.png')
plt.show()

plt.figure()
CT_X, CT_Y, CT_Z = posteriorPlot(50, 40, w_m1, w_c1, w_m2, w_c2, w_p1, w_p2)
plt.contour(CT_X, CT_Y, CT_Z, 3)
plt.title('W Posterior Probability')
plt.savefig('Lab3/w_posterior.png')
plt.show()
#%%
