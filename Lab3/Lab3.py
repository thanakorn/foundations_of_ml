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
#plt.savefig('Lab3/y_scatter.png')
plt.show()

plt.figure()
CT_X, CT_Y, CT_Z = posteriorPlot(50, 40, y_m1, y_c1, y_m2, y_c2, y_p1, y_p2)
plt.contour(CT_X, CT_Y, CT_Z, 3)
plt.title('Y Posterior Probability')
# plt.savefig('Lab3/y_posterior.png')
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
#plt.savefig('Lab3/z_scatter.png')
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
# plt.savefig('Lab3/w_scatter.png')
plt.show()

plt.figure()
CT_X, CT_Y, CT_Z = posteriorPlot(50, 40, w_m1, w_c1, w_m2, w_c2, w_p1, w_p2)
plt.contour(CT_X, CT_Y, CT_Z, 3)
plt.title('W Posterior Probability')
# plt.savefig('Lab3/w_posterior.png')
plt.show()
#%%2. FLDA and ROC
num_data = 200
X = np.random.randn(num_data,2)

m1 = np.array([0,3])
m2 = np.array([3,2.5])
c1 = np.array([[2,1],[1,2]])
c2 = np.array([[2,1],[1,2]])

F1 = np.linalg.cholesky(c1)
F2 = np.linalg.cholesky(c2)

A = (X @ F1) + m1
B = (X @ F2) + m2

plt.figure()
plt.scatter(Y1[:,0], Y1[:,1], color = 'r', label = 'A')
plt.scatter(Y2[:,0], Y2[:,1], color = 'g', label = 'B')
CT_X, CT_Y, CT_Z = twoDGaussianPlot(50, 40, m1, c1)
plt.contour(CT_X, CT_Y, CT_Z, 3, colors = 'r')
CT_X, CT_Y, CT_Z = twoDGaussianPlot(50, 40, m2, c2)
plt.contour(CT_X, CT_Y, CT_Z, 3, colors = 'g')
plt.legend()
# plt.savefig('Lab3/flda_roc_scatter.png')
plt.show()

w = np.linalg.inv((c1 + c2)) @ (m2 - m1)
A_prj = A @ w
B_prj = B @ w
plt.figure()
plt.hist(A_prj, bins = 30, color = 'r', label = 'A')
plt.hist(B_prj, bins = 30, color = 'g', label = 'B')
# plt.savefig('Lab3/flda_roc_prj_hist.png')
plt.show()

# Calculating ROC
pmin = np.min(np.array(np.min(A_prj), np.min(B_prj)))
pmax = np.max(np.array(np.max(A_prj), np.max(B_prj)))
print('Min : ', pmin)
print('Max : ', pmax)

num_roc_points = 50
thresholds = np.linspace(pmin, pmax, num_roc_points)
roc = np.zeros((num_roc_points,2))

for i in range(len(thresholds)):
    threshold = thresholds[i]
    true_pos = len(B_prj[B_prj > threshold]) * 100 / len(B_prj)
    false_pos = len(A_prj[A_prj > threshold]) * 100 / len(A_prj)
    roc[i,:] = [true_pos, false_pos]

# Plot ROC curve
plt.figure()
plt.plot(roc[:,1], roc[:,0], c='m')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC')
plt.grid(True)
# plt.savefig('Lab3/roc.png')
plt.show()
roc_area = np.trapz(roc[:,0], x = roc[:,1])
print('FLDA ROC Area : ', roc_area)

#%% 2.1 Random direction
random_dir = np.random.rand(2)
A_prj_2 = A @ random_dir
B_prj_2 = B @ random_dir
plt.figure()
plt.hist(A_prj_2, bins = 30, color = 'r', label = 'A')
plt.hist(B_prj_2, bins = 30, color = 'g', label = 'B')
#plt.savefig('Lab3/scatter_random_dir.png')
plt.show()

pmin = np.min(np.array(np.min(A_prj_2), np.min(B_prj_2)))
pmax = np.max(np.array(np.max(A_prj_2), np.max(B_prj_2)))
print('Min : ', pmin)
print('Max : ', pmax)

num_roc_points = 50
thresholds = np.linspace(pmin, pmax, num_roc_points)
roc_random = np.zeros((num_roc_points,2))

for i in range(len(thresholds)):
    threshold = thresholds[i]
    true_pos = len(B_prj_2[B_prj_2 > threshold]) * 100 / len(B_prj_2)
    false_pos = len(A_prj_2[A_prj_2 > threshold]) * 100 / len(A_prj_2)
    roc_random[i,:] = [true_pos, false_pos]

# Plot ROC curve
plt.figure()
plt.plot(roc_random[:,1], roc_random[:,0], c='m')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC')
plt.grid(True)
#plt.savefig('Lab3/roc_random_dir.png')
plt.show()
random_roc_area = np.trapz(roc_random[:,0], x = roc_random[:,1])
print('Random Direction ROC Area : ', random_roc_area)
#%% 2.2 Direction connecting means
mean_dir = (m2 - m1)
A_prj_3 = A @ mean_dir
B_prj_3 = B @ mean_dir
plt.figure()
plt.hist(A_prj_3, bins = 30, color = 'r', label = 'A')
plt.hist(B_prj_3, bins = 30, color = 'g', label = 'B')
# plt.savefig('Lab3/hist_mean.png')
plt.show()

pmin = np.min(np.array(np.min(A_prj_3), np.min(B_prj_3)))
pmax = np.max(np.array(np.max(A_prj_3), np.max(B_prj_3)))
print('Min : ', pmin)
print('Max : ', pmax)

num_roc_points = 50
thresholds = np.linspace(pmin, pmax, num_roc_points)
roc_mean = np.zeros((num_roc_points,2))

for i in range(len(thresholds)):
    threshold = thresholds[i]
    true_pos = len(B_prj_3[B_prj_3 > threshold]) * 100 / len(B_prj_3)
    false_pos = len(A_prj_3[A_prj_3 > threshold]) * 100 / len(A_prj_3)
    roc_mean[i,:] = [true_pos, false_pos]

# Plot ROC curve
plt.figure()
plt.plot(roc_mean[:,1], roc_mean[:,0], c='m')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC')
plt.grid(True)
# plt.savefig('Lab3/roc_mean.png')
plt.show()
mean_roc_area = np.trapz(roc_mean[:,0], x = roc_mean[:,1])
print('Mean Direction ROC Area : ', mean_roc_area)

#%% Mahalanobis Distance
def mahalanobis_dist(x, mean, cov):
    cov_inv = np.linalg.inv(cov)
    return np.sqrt((x - mean).T @ cov_inv @ (x - mean))

#%% Euclidean Distance
def euclidean_dist(x, mean):
    return np.sqrt((x - mean).T @ (x - mean))

# %% Distance to mean classifier
# Y correlated
# W uncorrelated

data = np.concatenate((W1, W2))
N = data.shape[0]
labels = np.concatenate((np.ones(num_data), -1 * np.ones(num_data)))
num_correct_mah = 0
num_correct_euc = 0
for i in range(N):
    x = data[i,:]
    euc_dist_pos = euclidean_dist(x, w_m1)
    euc_dist_neg = euclidean_dist(x, w_m2)
    mah_dist_pos = mahalanobis_dist(x, w_m1, w_c1)
    mah_dist_neg = mahalanobis_dist(x, w_m2, w_c2)
    # print("DistanceE to POS : ", euc_dist_pos)
    # print("DistanceE to NEG : ", euc_dist_neg)
    # print("DistanceM to POS : ", mah_dist_pos)
    # print("DistanceM to NEG : ", mah_dist_neg)
    euc_predict = 1 if euc_dist_pos <= euc_dist_neg else -1
    mah_predict = 1 if mah_dist_pos <= mah_dist_neg else -1
    if(euc_predict == labels[i]): num_correct_euc += 1
    if(mah_predict == labels[i]): num_correct_mah += 1

print("Euclidean Distance to Mean accuracy : ", 100 * (num_correct_euc / N))
print("Mahalanobis Distance to Mean accuracy : ", 100 * (num_correct_mah / N))

# %%
