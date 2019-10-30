#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, datasets, linear_model
from sklearn.linear_model import LinearRegression, Lasso
plt.rcParams.update(plt.rcParamsDefault)
#%% Linear Least Squares Regression
data, target = datasets.load_diabetes(True)
num_data, num_features = data.shape

# Visualizing
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
ax[0].hist(target, bins = 40)
ax[0].set_title('Distribution of Target')
ax[1].scatter(data[:,4], data[:,2])
ax[1].set_title('Scatter of 2 attributes')

# Using pseudo-inverse method
pseudo_inv_weights = np.linalg.inv(data.T @ data) @ data.T @ target
pseudo_inv_predict = data @ pseudo_inv_weights

# Using sklearn
linear_regression = LinearRegression()
linear_regression.fit(data, target)
sk_learn_predict = linear_regression.predict(data)

# Comparing accuracy
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
ax[0].scatter(target, pseudo_inv_predict, c = 'r')
ax[0].set_title('Pseudo-inverse method')
ax[1].scatter(target, sk_learn_predict, c = 'g')
ax[1].set_title('Scikit-learn method')

#%% Regularization

# Tikhanov
gamma = 0.5
reg_pseudo_inv_weights = np.linalg.inv((data.T @ data + gamma * np.eye(num_features))) @ data.T @ target

# Visualize the weights
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4))
ax[0].bar(range(0,num_features), pseudo_inv_weights)
ax[0].set_title('Pseudo-inverse solution')
ax[1].bar(range(0,num_features), reg_pseudo_inv_weights)
ax[1].set_title('Regularized pseudo-inverse solution')

# %%
