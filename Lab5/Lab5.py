# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, datasets, linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import normalize, scale
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
%matplotlib inline
plt.rcParams.update(plt.rcParamsDefault)

# %% Gaussian Function
def gaussian(x, u, sigma):
    return np.exp(-0.5 * np.linalg.norm(x - u) / sigma)

# %% Gaussian RBF model
diabetes = datasets.load_diabetes()
diabetes_data = diabetes.data
bias = np.ones((diabetes_data.shape[0], 1))
diabetes_data = np.hstack((diabetes_data, bias))
diabetes_target = diabetes.target

num_data, num_features = diabetes_data.shape

train_data, test_data, train_target, test_target = train_test_split(diabetes_data, diabetes_target, test_size=0.3)

# Design matrix
M = 200

# Basis function = distance between two random data
C = np.random.randn(M, num_features)
x1 = diabetes_data[np.floor(np.random.rand() * num_data).astype(int), :]
x2 = diabetes_data[np.floor(np.random.rand() * num_data).astype(int), :]
sigma = np.linalg.norm(x1 - x2)

U_train = np.zeros((train_data.shape[0], M))
for i in range(train_data.shape[0]):
    for j in range(M):
        U_train[i,j] = gaussian(train_data[i,:], C[j,:], sigma)

U_test = np.zeros((test_data.shape[0], M))
for i in range(test_data.shape[0]):
    for j in range(M):
        U_test[i,j] = gaussian(test_data[i,:], C[j,:], sigma)

# Finding coefficients
w = np.linalg.inv(U_train.T @ U_train) @ U_train.T @ train_target

diabetes_predict_train = U_train @ w
diabetes_predict_test = U_test @ w

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,3))
ax[0].grid(True)
ax[0].set_title('Training Set')
ax[0].set_xlabel('Actual')
ax[0].set_ylabel('Predict')
ax[0].scatter(train_target, diabetes_predict_train)
ax[0].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(train_target, diabetes_predict_train), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[1].grid(True)
ax[1].set_title('Test Set')
ax[1].set_xlabel('Actual')
ax[1].set_ylabel('Predict')
ax[1].scatter(test_target, diabetes_predict_test, color='r')
ax[1].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(test_target, diabetes_predict_test), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
# plt.savefig('Lab5/rbf_original', bbox_inches ='tight')

# %% RBF Improvement
# Normalized features
normalized_diabetes_data = np.copy(diabetes_data)
for i in range(num_features - 1):
    feature = diabetes_data[:,i]
    normalized_feature = scale(feature, axis = 0)
    normalized_diabetes_data[:,i] = normalized_feature

# Adjusting sigma based on several data points
num_data_points = 50
random_point_indices = np.random.randint(0, num_data, num_data_points)
random_points_1, random_points_2 = np.split(normalized_diabetes_data[random_point_indices], 2)
distances = np.zeros(int(num_data_points / 2))
for i in range(int(num_data_points / 2)):
    distances[i] = np.linalg.norm(random_points_1[i] - random_points_2[i])
avg_sigma = distances.mean()
# Adjusting location of basis function
C_cluster = KMeans(n_clusters=M).fit(normalized_diabetes_data).cluster_centers_

# Finding new coefficients and predict
M_new = 20
train_data, test_data, train_target, test_target = train_test_split(normalized_diabetes_data, diabetes_target, test_size=0.3)

num_data_train = train_data.shape[0]
num_data_test = test_data.shape[0]

U_train = np.zeros((num_data_train, M_new))
for i in range(num_data_train):
    for j in range(M_new):
        U_train[i,j] = gaussian(train_data[i,:], C_cluster[j,:], avg_sigma)

U_test = np.zeros((num_data_test, M_new))
for i in range(num_data_test):
    for j in range(M_new):
        U_test[i,j] = gaussian(test_data[i,:], C_cluster[j,:], avg_sigma)


w_new = np.linalg.inv(U_train.T @ U_train) @ U_train.T @ train_target

diabetes_predict_train = U_train @ w_new
diabetes_predict_test = U_test @ w_new
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,3))
ax[0].grid(True)
ax[0].set_title('Training Set')
ax[0].set_xlabel('Actual')
ax[0].set_ylabel('Predict')
ax[0].scatter(train_target, diabetes_predict_train)
ax[0].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(train_target, diabetes_predict_train), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[1].grid(True)
ax[1].set_title('Test Set')
ax[1].set_xlabel('Actual')
ax[1].set_ylabel('Predict')
ax[1].scatter(test_target, diabetes_predict_test, color='r')
ax[1].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(test_target, diabetes_predict_test), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
# plt.savefig('Lab5/rbf_improved', bbox_inches='tight')

# %% 10-Fold Cross-validation
K = 10
M = 20
C = KMeans(n_clusters=M).fit(normalized_diabetes_data).cluster_centers_
kfold = KFold(n_splits=K)
rbf_errors = np.zeros(K)
lr_errors = np.zeros(K)
round = 0
for train_indices, test_indices in kfold.split(normalized_diabetes_data):
    train_data = normalized_diabetes_data[train_indices]
    train_target = diabetes_target[train_indices]
    test_data = normalized_diabetes_data[test_indices]
    test_target = diabetes_target[test_indices]
    U_train = np.zeros((len(train_indices), M))
    U_test = np.zeros((len(test_data), M))
    for i in range(len(train_indices)):
        for j in range(M):
            U_train[i,j] = gaussian(train_data[i,:], C[j,:], avg_sigma)
    for i in range(len(test_indices)):
        for j in range(M):
            U_test[i,j] = gaussian(test_data[i,:], C[j,:], avg_sigma)

    w_rbf = np.linalg.inv(U_train.T @ U_train) @ U_train.T @ train_target
    w_lr = np.linalg.inv(train_data.T @ train_data + (0.2 * np.eye(num_features))) @ train_data.T @ train_target
    predict_train = U_train @ w_rbf
    predict_rbf = U_test @ w_rbf
    predict_lr = test_data @ w_lr
    rbf_err = mean_squared_error(test_target, predict_rbf)
    lr_err = mean_squared_error(test_target, predict_lr)
    rbf_errors[round] = rbf_err
    lr_errors[round] = lr_err
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,3))
    ax[0].grid(True)
    ax[0].set_title('Linear Regression')
    ax[0].set_xlabel('Actual')
    ax[0].set_ylabel('Predict')
    ax[0].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % lr_err, weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
    ax[0].scatter(test_target, predict_lr)
    ax[1].grid(True)
    ax[1].set_title('RBF')
    ax[1].set_xlabel('Actual')
    ax[1].set_ylabel('Predict')
    ax[1].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % rbf_err, weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
    ax[1].scatter(test_target, predict_rbf, color='r')
    # plt.savefig(f'Lab5/train_test_perf_%d' % (round+1))

    fig, ax = plt.subplots(figsize=(3,3))
    ax.grid(True)
    ax.boxplot([test_target, predict_lr, predict_rbf], labels=['Actual', 'Linear\nRegression', 'RBF'])
    # plt.savefig(f'Lab5/result_dist_%d' % (round+1), bbox_inches='tight')
    round += 1

#%%
fig, ax = plt.subplots(figsize=(3,3))
ax.set_title('Errors on Test set')
ax.grid(True)
ax.boxplot([lr_errors, rbf_errors], labels=['Linear\nRegression', 'RBF'])
# plt.savefig('Lab5/error_compare', bbox_inches='tight')

# %% Compare with scikit-learn RBF
train_data, test_data, train_target, test_target = train_test_split(normalized_diabetes_data, diabetes_target, test_size=0.3)
svr_rbf = SVR(kernel='rbf', gamma='scale', epsilon=100.0)
svr_rbf.fit(train_data, train_target)
predict_svr_train = svr_rbf.predict(train_data)
predict_svr_test = svr_rbf.predict(test_data)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
ax[0].set_title('Training Set')
ax[0].set_ylim((0,300))
ax[0].scatter(train_target, predict_svr_train)
ax[0].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(train_target, predict_svr_train), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[0].transAxes)
ax[1].set_title('Test Set')
ax[1].set_ylim((0,300))
ax[1].scatter(test_target, predict_svr_test, color='r')
ax[1].text(0.75, 0.1, f'Mean Sqr Error = %.2f' % mean_squared_error(test_target, predict_svr_test), weight = 'bold', horizontalalignment='center',verticalalignment='center', transform=ax[1].transAxes)
plt.savefig('Lab5/sklearn_compare', bbox_inches='tight')
# %%
