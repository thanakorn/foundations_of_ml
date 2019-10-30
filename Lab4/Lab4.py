#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, datasets, linear_model
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
plt.rcParams.update(plt.rcParamsDefault)
#%% Linear Least Squares Regression
raw_data, target = datasets.load_diabetes(True)
num_data, num_features = raw_data.shape
data = raw_data
bias = np.ones((num_data, 1))
data = np.append(raw_data, bias, axis = 1)

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
linear_regression.fit(raw_data, target)
sk_learn_predict = linear_regression.predict(raw_data)

# Comparing accuracy
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
ax[0].scatter(target, pseudo_inv_predict, c = 'r')
ax[0].set_title('Pseudo-inverse method')
ax[1].scatter(target, sk_learn_predict, c = 'g')
ax[1].set_title('Scikit-learn method')

#%% Regularization

# Tikhanov Regularization
gamma = 0.5
reg_pseudo_inv_weights = np.linalg.inv((data.T @ data + gamma * np.eye(num_features + 1))) @ data.T @ target

# Lasso Regularization
lasso = Lasso(alpha = 0.2)
lasso.fit(raw_data, target)
lasso_predict = lasso.predict(raw_data)

# Visualize the weights
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 4))
ax[0].bar(range(0,num_features + 1), pseudo_inv_weights)
ax[0].set_title('Pseudo-inverse solution')
ax[0].set_ylim(np.min(pseudo_inv_weights), np.max(pseudo_inv_weights))
ax[0].set_xlim(0, num_features + 1)
ax[1].bar(range(0,num_features + 1), reg_pseudo_inv_weights)
ax[1].set_title('Regularized pseudo-inverse solution')
ax[1].set_ylim(np.min(pseudo_inv_weights), np.max(pseudo_inv_weights))
ax[1].set_xlim(0, num_features + 1)
ax[2].bar(range(0,len(lasso.coef_)), lasso.coef_)
ax[2].set_title('Lasso solution')
ax[2].set_ylim(np.min(pseudo_inv_weights), np.max(pseudo_inv_weights))
ax[2].set_xlim(0, num_features + 1)

# Visualize Regularization Path
_, _, coefs = linear_model.lars_path(raw_data, target, method = 'lasso', verbose = True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]
plt.figure()
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyles='dashed')
plt.xlabel('coef/max(coef)')

#%% Solubility
raw_solubility = pd.read_excel('/home/tpanyapiang/git/MSc/foundations_of_ml/Lab4/Husskonen_Solubility_Features.xlsx', verbose=False) # Load data from file

#%% Finding weights
num_data, num_features = raw_solubility.shape
attribute_names = raw_solubility.columns
solubility = np.append(raw_solubility[attribute_names[5:len(attribute_names)]].values, np.ones((num_data, 1)), axis=1)

target = raw_solubility['LogS.M.'].values
plt.figure()
plt.hist(target, bins=40)

solubility_train, solubility_test, target_train, target_test = train_test_split(solubility, target, test_size = 0.3)

#Regularized Regression
gamma = 2.3 * np.eye(len(attribute_names[5:len(attribute_names)]) + 1)
weights = np.linalg.inv(solubility_train.T @ solubility_train + gamma) @ solubility_train.T @ target_train
predict_train = solubility_train @ weights
predict_test = solubility_test @ weights

#Plot prediction
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
ax[0].scatter(target_train, predict_train, c='r')
ax[0].grid(True)
ax[0].set_title('Training Set Accuracy')
ax[1].scatter(target_test, predict_test, c='g')
ax[1].grid(True)
ax[1].set_title('Test Set Accuracy')