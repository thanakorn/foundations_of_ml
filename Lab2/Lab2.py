#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
Y = np.concatenate((Y1, Y2))
label_positive = np.ones(num_data_per_class)
label_negative = -1 * np.ones(num_data_per_class)
labels = np.concatenate((label_positive, label_negative))

sample_indices = np.random.permutation(num_data_per_class * 2)
shuffle_samples = Y[sample_indices,:]
shuffle_labels = labels[sample_indices]

train_samples = shuffle_samples[0:num_data_per_class,:]
train_labels = shuffle_labels[0:num_data_per_class]
test_samples = shuffle_samples[num_data_per_class:2*num_data_per_class,:]
test_labels = shuffle_labels[num_data_per_class:2*num_data_per_class]
print('Training data : ', train_samples.shape)
print('Test data : ', test_samples.shape)
#%% Calculate accuracy
def PercentCorrect(Inputs, targets, weights):
    N = len(targets)
    num_correct = 0
    for i in range(N):
        input = Inputs[i,:]
        if(targets[i] * np.dot(input, weights) > 0):
            num_correct += 1
    return (num_correct / N) * 100

#%% Testing with benchmark datasets
iris_data = pd.read_csv('/home/tpanyapiang/git/MSc/foundations_of_ml/Lab2/iris.data',
                        names=['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'class'])
wine_data = pd.read_csv('/home/tpanyapiang/git/MSc/foundations_of_ml/Lab2/wine.data',
                        names=['class','alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','OD280_OD315','proline'])
