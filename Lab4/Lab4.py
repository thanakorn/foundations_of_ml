#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, datasets, linear_model
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
%matplotlib inline
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
# plt.savefig('Lab4/diabetes_data.png')

# Using pseudo-inverse method
pseudo_inv_weights = np.linalg.inv(data.T @ data) @ data.T @ target
pseudo_inv_predict = data @ pseudo_inv_weights

# Using sklearn
linear_regression = LinearRegression()
linear_regression.fit(data, target)
sk_learn_predict = linear_regression.predict(data)

# Comparing accuracy
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
ax[0].scatter(target, pseudo_inv_predict)
ax[0].set_title('Pseudo-inverse method')
ax[0].set_ylabel('Predict')
ax[0].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(pseudo_inv_predict, target), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_xlabel('Actual')
ax[1].scatter(target, sk_learn_predict, c = 'r')
ax[1].set_title('Scikit-learn')
ax[1].set_ylabel('Predict')
ax[1].set_xlabel('Actual')
ax[1].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(sk_learn_predict, target), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
# plt.savefig('Lab4/diabetes_accuracy.png', bbox = 'tight')

#%% Regularization
# Tikhanov Regularization
gamma = 0.5
reg_pseudo_inv_weights = np.linalg.inv((data.T @ data + gamma * np.eye(num_features + 1))) @ data.T @ target

# Visualize the weights
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4))
ax[0].bar(range(0,num_features + 1), pseudo_inv_weights)
ax[0].set_title('Pseudo-inverse solution')
ax[0].set_ylim(np.min(pseudo_inv_weights), np.max(pseudo_inv_weights))
ax[0].set_xlim(0, num_features + 1)
ax[1].bar(range(0,num_features + 1), reg_pseudo_inv_weights)
ax[1].set_title('Regularized pseudo-inverse solution')
ax[1].set_ylim(np.min(pseudo_inv_weights), np.max(pseudo_inv_weights))
ax[1].set_xlim(0, num_features + 1)
# plt.savefig('Lab4/reg_weights_compare.png')

gamma_2 = 0.2
reg_pseudo_inv_weights_2 = np.linalg.inv((data.T @ data + gamma_2 * np.eye(num_features + 1))) @ data.T @ target
gamma_3 = 1.0
reg_pseudo_inv_weights_3 = np.linalg.inv((data.T @ data + gamma_3 * np.eye(num_features + 1))) @ data.T @ target
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 4))
ax[0].grid(True)
ax[0].bar(range(0,len(reg_pseudo_inv_weights_2)), reg_pseudo_inv_weights_2)
ax[0].set_title('γ = 0.2')
ax[0].set_ylim(np.min(reg_pseudo_inv_weights), np.max(reg_pseudo_inv_weights))
ax[0].set_xlim(0, num_features + 1)
ax[1].grid(True)
ax[1].bar(range(0,len(reg_pseudo_inv_weights)), reg_pseudo_inv_weights)
ax[1].set_title('γ = 0.5')
ax[1].set_ylim(np.min(reg_pseudo_inv_weights), np.max(reg_pseudo_inv_weights))
ax[1].set_xlim(0, num_features + 1)
ax[2].grid(True)
ax[2].bar(range(0,len(reg_pseudo_inv_weights_3)), reg_pseudo_inv_weights_3)
ax[2].set_title('γ = 1.0')
ax[2].set_ylim(np.min(reg_pseudo_inv_weights), np.max(reg_pseudo_inv_weights))
ax[2].set_xlim(0, num_features + 1)
# plt.savefig('Lab4/ridge_weights_compare.png')

#%% Sparse Regression
lasso_1 = Lasso(alpha = 0.2)
lasso_2 = Lasso(alpha = 0.5)
lasso_3 = Lasso(alpha = 1.0)
lasso_1.fit(raw_data, target)
lasso_2.fit(raw_data, target)
lasso_3.fit(raw_data, target)
lasso_1_predict = lasso_1.predict(raw_data)
lasso_2_predict = lasso_2.predict(raw_data)
lasso_3_predict = lasso_3.predict(raw_data)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 4))
ax[0].grid(True)
ax[0].bar(range(0,len(lasso_1.coef_)), lasso_1.coef_)
ax[0].set_title('γ = 0.2')
ax[0].set_ylim(np.min(lasso_1.coef_), np.max(lasso_1.coef_))
ax[0].set_xlim(0, num_features + 1)
ax[1].grid(True)
ax[1].bar(range(0,len(lasso_2.coef_)), lasso_2.coef_)
ax[1].set_title('γ = 0.5')
ax[1].set_ylim(np.min(lasso_1.coef_), np.max(lasso_1.coef_))
ax[1].set_xlim(0, num_features + 1)
ax[2].grid(True)
ax[2].bar(range(0,len(lasso_3.coef_)), lasso_3.coef_)
ax[2].set_title('γ = 1.0')
ax[2].set_ylim(np.min(lasso_1.coef_), np.max(lasso_1.coef_))
ax[2].set_xlim(0, num_features + 1)
plt.savefig('Lab4/lasso_compare.png')

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 4))
ax[0].grid(True)
ax[0].scatter(lasso_1_predict, target)
ax[0].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(lasso_1_predict, target), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_title('γ = 0.2')
ax[1].grid(True)
ax[1].scatter(lasso_2_predict, target)
ax[1].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(lasso_2_predict, target), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
ax[1].set_title('γ = 0.5')
ax[2].grid(True)
ax[2].scatter(lasso_3_predict, target)
ax[2].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(lasso_3_predict, target), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[2].transAxes)
ax[2].set_title('γ = 1.0')
plt.savefig('Lab4/lasso_error_compare.png')

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4))
ax[0].scatter(raw_data[:,8], raw_data[:,0])
ax[0].set_xlabel('Attribute #8')
ax[0].set_ylabel('Attribute #0')
ax[1].scatter(raw_data[:,8], raw_data[:,4])
ax[1].set_xlabel('Attribute #8')
ax[1].set_ylabel('Attribute #4')
plt.savefig('Lab4/correlation.png')

# Visualize Regularization Path
_, _, coefs = linear_model.lars_path(raw_data, target, method = 'lasso', verbose = True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]
plt.figure()
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyles='dashed')
plt.xlabel('coef/max(coef)')
plt.ylabel('coef')
plt.title('Regularization Path')
# plt.savefig('Lab4/reg_path.png')

#%% Solubility
raw_solubility = pd.read_excel('/home/tpanyapiang/git/MSc/foundations_of_ml/Lab4/Husskonen_Solubility_Features.xlsx', verbose=False) # Load data from file

#%% Finding weights
num_data, num_features = raw_solubility.shape
attribute_names = raw_solubility.columns
solubility = np.append(raw_solubility[attribute_names[5:len(attribute_names)]].values, np.ones((num_data, 1)), axis=1)
target = raw_solubility['LogS.M.'].values
plt.figure()
plt.hist(target, bins=40)
# plt.savefig('Lab4/solubility_target_hist.png')

solubility_train, solubility_test, target_train, target_test = train_test_split(solubility, target, test_size = 0.3)

# Regularized Regression using Tikhanov
gamma = 2.3 * np.eye(len(attribute_names[5:len(attribute_names)]) + 1)
weights = np.linalg.inv(solubility_train.T @ solubility_train + gamma) @ solubility_train.T @ target_train
tikhanov_predict_train = solubility_train @ weights
tikhanov_predict_test = solubility_test @ weights

# Regularized Regressaion using Lasso
lasso = Lasso(alpha=0.2)
lasso.fit(solubility_train, target_train)
lasso_predict_train = lasso.predict(solubility_train)
lasso_predict_test = lasso.predict(solubility_test)

lasso_2 = Lasso(alpha=1.0)
lasso_2.fit(solubility_train, target_train)
lasso_2_predict_train = lasso_2.predict(solubility_train)
lasso_2_predict_test = lasso_2.predict(solubility_test)

lasso_3 = Lasso(alpha=3.0)
lasso_3.fit(solubility_train, target_train)
lasso_3_predict_train = lasso_3.predict(solubility_train)
lasso_3_predict_test = lasso_3.predict(solubility_test)

# %%
#Plot prediction
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
ax[0].scatter(target_train, tikhanov_predict_train)
ax[0].grid(True)
ax[0].set_title('L2 Regularization Training Accuracy')
ax[0].text(0.75, 0.1, f'Total Sqr Error = %.2f' % np.sum((tikhanov_predict_train - target_train)**2), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_xlim(-12,3)
ax[0].set_ylim(-12,3)
ax[1].scatter(target_test, tikhanov_predict_test, c='r')
ax[1].grid(True)
ax[1].set_title('L2 Regularization Test Accuracy')
ax[1].text(0.75, 0.1, f'Total Sqr Error = %.2f' % np.sum((tikhanov_predict_test - target_test)**2), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
ax[1].set_xlim(-12,3)
ax[1].set_ylim(-12,3)
# plt.savefig('Lab4/tikhanov_solubility_acc.png')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,4))
ax[0].scatter(target_test, lasso_predict_test)
ax[0].grid(True)
ax[0].set_title('γ = 0.2')
ax[0].text(0.75, 0.1, f'Total Sqr Error = %.2f' % np.sum((lasso_predict_test - target_test)**2), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_xlim(-12,3)
ax[0].set_ylim(-12,3)
ax[1].scatter(target_test, lasso_predict_test)
ax[1].grid(True)
ax[1].set_title('γ = 1.0')
ax[1].text(0.75, 0.1, f'Total Sqr Error = %.2f' % np.sum((lasso_2_predict_test - target_test)**2), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
ax[1].set_xlim(-12,3)
ax[1].set_ylim(-12,3)
ax[2].scatter(target_test, lasso_predict_test)
ax[2].grid(True)
ax[2].set_title('γ = 3.0')
ax[2].text(0.75, 0.1, f'Total Sqr Error = %.2f' % np.sum((lasso_3_predict_test - target_test)**2), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[2].transAxes)
ax[2].set_xlim(-12,3)
ax[2].set_ylim(-12,3)
# plt.savefig('Lab4/lasso_solubility_acc.png')

# %%
non_zero_lasso_weights = lasso.coef_[lasso.coef_ > 0]
non_zero_lasso_2_weights = lasso_2.coef_[lasso_2.coef_ > 0]
non_zero_lasso_3_weights = lasso_3.coef_[lasso_3.coef_ > 0]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,4))
ax[0].set_title('γ = 0.2')
ax[0].grid(True)
ax[0].bar(range(0,len(non_zero_lasso_weights)), non_zero_lasso_weights)
ax[0].set_ylim(np.min(non_zero_lasso_weights), np.max(non_zero_lasso_weights))
ax[0].set_xlim(0, len(non_zero_lasso_weights))
ax[1].set_title('γ = 1.0')
ax[1].grid(True)
ax[1].bar(range(0,len(non_zero_lasso_2_weights)), non_zero_lasso_2_weights)
ax[1].set_ylim(np.min(non_zero_lasso_2_weights), np.max(non_zero_lasso_2_weights))
ax[1].set_xlim(0, len(non_zero_lasso_2_weights))
ax[2].set_title('γ = 3.0')
ax[2].grid(True)
ax[2].bar(range(0,len(non_zero_lasso_3_weights)), non_zero_lasso_3_weights)
ax[2].set_ylim(np.min(non_zero_lasso_3_weights), np.max(non_zero_lasso_3_weights))
ax[2].set_xlim(0, len(non_zero_lasso_3_weights))
# plt.savefig('Lab4/lasso_solubility_coeff.png')


# %% Compare result
training_set, training_target = (solubility[0:161,:], target[0:161])
test_set, test_target = (solubility[161:211,:], target[161:211])
lasso = Lasso(alpha=0.5)
lasso.fit(training_set, training_target)
lasso_predict_train = lasso.predict(training_set)
lasso_predict_test = lasso.predict(test_set)
error_train = np.sum((lasso_predict_train - training_target)**2)
error_test = np.sum((lasso_predict_test - test_target)**2)

huuskonen_result = pd.read_csv('/home/tpanyapiang/git/MSc/foundations_of_ml/Lab4/huuskonen_result.csv', sep=',', header=None, usecols=[2,3],names=['obs','predict'])
huuskonen_error = huuskonen_result['obs'].values - huuskonen_result['predict'].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
ax[0].scatter(test_target, lasso_predict_test)
ax[0].grid(True)
ax[0].set_title('Lasso Regression')
ax[0].text(0.75, 0.1, f'Total Sqr Error = %.2f' % error_test, weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_xlim(-12,3)
ax[0].set_ylim(-12,3)
ax[1].scatter(huuskonen_result['obs'], huuskonen_result['predict'])
ax[1].grid(True)
ax[1].set_title('Artificial Neural Network')
ax[1].text(0.75, 0.1, f'Total Sqr Error = %.2f' % np.sum(huuskonen_error**2), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
ax[1].set_xlim(-12,3)
ax[1].set_ylim(-12,3)
plt.savefig('Lab4/lasso_ann_compare.png')
# %%
