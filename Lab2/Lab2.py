#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#%% Generate Samples
num_data_per_class = 200

m1 = [[0,5]]
m2 = [[5,0]]
C  = [[2,1], [1,2]]

A = np.linalg.cholesky(C)

X1 = np.random.randn(num_data_per_class, 2)
Y1 = (X1 @ A) + m1

X2 = np.random.randn(num_data_per_class, 2)
Y2 = (X2 @ A) + m2

plt.figure(figsize=(3,3))
plt.scatter(Y1[:,0], Y1[:,1], c = 'r', label = 'Y1')
plt.scatter(Y2[:,0], Y2[:,1], c = 'b', label = 'Y2')
plt.legend()
plt.savefig('scatter.png')
#%% Generate training set and testing set
samples = np.concatenate((Y1, Y2))
label_positive = np.ones(num_data_per_class)
label_negative = np.ones(num_data_per_class)
labels = np.concatenate((label_positive, label_negative))

sample_indices = np.random.permutation(num_data_per_class * 2)
train_samples = samples[sample_indices, :]
train_labels = labels[sample_indices]