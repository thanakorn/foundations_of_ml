# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, datasets, linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_spd_matrix
%matplotlib inline
plt.rcParams.update(plt.rcParamsDefault)

# %% Generate samples from Gaussian Density
def genGaussianSamples(N, m, C):
    A = np.linalg.cholesky(C)
    U = np.random.randn(N, 2)
    return U @ A.T + m

# %% Contour plot of a Gaussian Density
def gauss2D(x, m, C):
    Ci = np.linalg.inv(C)
    dC = np.linalg.det(C)
    num = np.exp(-0.5 * np.dot((x -  m).T, np.dot(Ci, (x-m))))
    den = 2 * np.pi * dC
    return num / den

def twoDGaussianPlot(nx, ny, m, C):
    x = np.linspace(-5, 8, nx)
    y = np.linspace(-3, 12, ny)
    X, Y = np.meshgrid(x, y, indexing = 'ij')

    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            Z[i,j] = gauss2D(xvec, m, C)

    return X, Y, Z

# %% K-Mean clustering method
# def kmean_clustering(data, K):
def kmean_clustering(data, K, ax):
    num_data = data.shape[0]
    num_attribute = data.shape[1]
    # initial means
    cluster_centers = data[np.random.randint(0, num_data, size = K),:]
    # cluster_centers = np.random.rand(K, num_attribute)
    # cluster_centers = np.array([[-6,16], [1, 5], [6,16]])
    # cluster_centers = np.array([[-6,16], [1, 5], [2,2.5]])
    distance_to_m = np.zeros((num_data, K))
    cluster = np.zeros(num_data)

    while True:
        for i in range(K):
            ax.scatter(cluster_centers[i,0], cluster_centers[i,1], c='b', s=20, alpha=0.75)

        new_cluster = np.zeros(num_data)
        for i in range(num_data):
            for j in range(K):
                distance_to_m[i,j] = np.linalg.norm(data[i,:] - cluster_centers[j,:])
            new_cluster[i] = np.argmin(distance_to_m[i,:])

        if np.array_equal(cluster, new_cluster):
            break
        else:
            cluster = new_cluster
            for i in range(K):
                cluster_members = data[np.argwhere(cluster == i)]
                if(cluster_members.size > 0):
                    new_m = np.mean(cluster_members, 0) # Adjust m
                    cluster_centers[i,:] = new_m
    return cluster_centers
# %% Gen samples
# W = np.random.rand(3)
W = np.array([0.3, 0.25, 0.45])
W = W / np.sum(W)
n_data = np.floor(W * 1000).astype(int)
M = np.array([[2, 1], [-2, 1], [0, 5]])
C = np.zeros((3, 2, 2))
for i in range(3):
    C[i,:,:] = make_spd_matrix(2)

X = genGaussianSamples(n_data[0], M[0,:], C[0,:,:])
Y = genGaussianSamples(n_data[1], M[1,:], C[1,:,:])
Z = genGaussianSamples(n_data[2], M[2,:], C[2,:,:])

fig, ax = plt.subplots()
ax.grid(True)
ax.scatter(X[:,0], X[:,1], s=15)
ax.scatter(Y[:,0], Y[:,1], s=15, c='r')
ax.scatter(Z[:,0], Z[:,1], s=15, c='g')
Xp, Yp, Zp = twoDGaussianPlot(100, 100, M[0], C[0,:,:])
ax.contour(Xp, Yp, Zp, 3)
Xp, Yp, Zp = twoDGaussianPlot(100, 100, M[1], C[1,:,:])
ax.contour(Xp, Yp, Zp, 3)
Xp, Yp, Zp = twoDGaussianPlot(100, 100, M[2], C[2,:,:])
ax.contour(Xp, Yp, Zp, 3)
# plt.savefig('Lab6/distribution-balance.jpg', bbox='tight')

# %%
data = np.concatenate((X,Y,Z))
K = 3
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
ax[0].set_title('Cluster Centers')
ax[0].scatter(data[:,0], data[:,1], s=15, c='grey', alpha=0.75)
centers = kmean_clustering(data, K, ax[0])
ax[0].scatter(centers[:,0], centers[:,1], s=50, c='r')
sk_centers = KMeans(n_clusters=K).fit(data).cluster_centers_
ax[1].set_title('Scikit-learn Cluster Centers')
ax[1].scatter(data[:,0], data[:,1], s=15, c='grey', alpha=0.75)
ax[1].scatter(sk_centers[:,0], sk_centers[:,1], s=50, c='r')
# plt.savefig('Lab6/sklean-compare-balance.jpg', bbox='tight')

# %% Analyze effects of K
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))
centers = kmean_clustering(data, 2)
ax[0].set_title('K = 2')
ax[0].scatter(data[:,0], data[:,1], s=15, c='grey', alpha=0.75)
ax[0].scatter(centers[:,0], centers[:,1], s=50, c='r')
centers = kmean_clustering(data, 4)
ax[1].set_title('K = 4')
ax[1].scatter(data[:,0], data[:,1], s=15, c='grey', alpha=0.75)
ax[1].scatter(centers[:,0], centers[:,1], s=50, c='r')
centers = kmean_clustering(data, 7)
ax[2].set_title('K = 7')
ax[2].scatter(data[:,0], data[:,1], s=15, c='grey', alpha=0.75)
ax[2].scatter(centers[:,0], centers[:,1], s=50, c='r')
# plt.savefig('Lab6/multiple_k.jpg')

# %% Elbow method
distorsions = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(6, 3))
plt.plot(range(2, 10), distorsions)
plt.ylabel('Average(distance to centroid$^2$)')
plt.xlabel('K')
plt.grid(True)
plt.title('Elbow Curve')
plt.savefig('Lab6/elbow.jpg', bbox='tight')

# %% Analyze effects of initial centers 1
fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1], s=15, c='grey', alpha=0.75)
centers = kmean_clustering(data, 3, ax)
ax.scatter(centers[:,0], centers[:,1], s=50, c='r')
# plt.savefig('Lab6/one_nearest_centroid.jpg')

# %% Analyze effects of initial centers 2
fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1], s=15, c='grey', alpha=0.75)
centers = kmean_clustering(data, 3, ax)
ax.scatter(centers[:,0], centers[:,1], s=50, c='r')
# plt.savefig('Lab6/one_furthest_centroid.jpg', bbox='tight')

# %% Hand postures dataset
hand_postures_raw = pd.read_csv('/home/tpanyapiang/git/MSc/foundations_of_ml/Lab6/Postures.csv',
                na_values='?').drop([0]).sample(frac=0.02)
hand_postures_labels = hand_postures_raw['Class']
# %% Data preproscessing
missing_columns = hand_postures_raw.columns[hand_postures_raw.isna().any()].tolist()
scaler = MinMaxScaler()
hand_postures = hand_postures_raw.drop(columns=['Class','User'] + missing_columns).to_numpy()
hand_postures = scaler.fit_transform(hand_postures)
hand_postures = hand_postures[:,0:3]
# %% Visualizing data
for i in hand_postures_raw['Class'].unique():
    posture = hand_postures[hand_postures_labels == i,:]
    fig = plt.figure(figsize=(18,3))
    fig.suptitle(f'Posture {i}')
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title('Marker 1')
    ax.scatter(posture[:,0], posture[:,1], posture[:,2])
    # ax = fig.add_subplot(1, 3, 2, projection='3d')
    # ax.set_title('Marker 2')
    # ax.scatter(posture[:,3], posture[:,4], posture[:,5], c='r')
    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    # ax.set_title('Marker 3')
    # ax.scatter(posture[:,6], posture[:,7], posture[:,8], c='g')

# %% Perform clustering
K = 5
centers = kmean_clustering(hand_postures, K)
sk_centers = KMeans(n_clusters=K).fit(hand_postures).cluster_centers_

# %%
fig = plt.figure(figsize=(18,3))
fig.suptitle(f'Cluster Centers')
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(0.0, 1.0)
ax.set_title('Marker 1')
ax.scatter(centers[:,0], centers[:,1], centers[:,2])
# ax = fig.add_subplot(1, 3, 2, projection='3d')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)
# ax.set_zlim(0.0, 1.0)
# ax.set_title('Marker 2')
# ax.scatter(centers[:,3], centers[:,4], centers[:,5], c='r')
# ax = fig.add_subplot(1, 3, 3, projection='3d')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)
# ax.set_zlim(0.0, 1.0)
# ax.set_title('Marker 3')
# ax.scatter(centers[:,6], centers[:,7], centers[:,8], c='g')

fig = plt.figure(figsize=(18,3))
fig.suptitle(f'Scikit-Learn Cluster Centers')
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(0.0, 1.0)
ax.set_title('Marker 1')
ax.scatter(sk_centers[:,0], sk_centers[:,1], sk_centers[:,2])
# ax = fig.add_subplot(1, 3, 2, projection='3d')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)
# ax.set_zlim(0.0, 1.0)
# ax.set_title('Marker 2')
# ax.scatter(sk_centers[:,3], sk_centers[:,4], sk_centers[:,5], c='r')
# ax = fig.add_subplot(1, 3, 3, projection='3d')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)
# ax.set_zlim(0.0, 1.0)
# ax.set_title('Marker 3')
# ax.scatter(sk_centers[:,6], sk_centers[:,7], sk_centers[:,8], c='g')

# %% Iris dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target
iris_desc = iris.feature_names

iris_0 = iris_data[iris_target == 0,:]
iris_1 = iris_data[iris_target == 1,:]
iris_2 = iris_data[iris_target == 2,:]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
ax[0].scatter(iris_0[:,0], iris_0[:,1])
ax[0].scatter(iris_1[:,0], iris_1[:,1])
ax[0].scatter(iris_2[:,0], iris_2[:,1])
ax[0].set_xlabel(iris_desc[0])
ax[0].set_ylabel(iris_desc[1])
ax[1].scatter(iris_0[:,2], iris_0[:,3])
ax[1].scatter(iris_1[:,2], iris_1[:,3])
ax[1].scatter(iris_2[:,2], iris_2[:,3])
ax[1].set_xlabel(iris_desc[2])
ax[1].set_ylabel(iris_desc[3])
# plt.savefig('Lab6/iris_dist.jpg', bbox='tight')

# %% Clustering
fig, ax = plt.subplots()
ax.scatter(iris_data[:,0], iris_data[:,1], s=15, c='grey')
ax.set_xlabel(iris_desc[0])
ax.set_ylabel(iris_desc[1])
iris_centers = kmean_clustering(iris_data, 3, ax)
ax.scatter(iris_centers[:,0], iris_centers[:,1], s=50, c='r', label='centroid')
# plt.savefig('Lab6/iris_cluster_1.jpg', bbox='tight')

fig, ax = plt.subplots()
ax.scatter(iris_data[:,2], iris_data[:,3], s=15, c='grey')
ax.set_xlabel(iris_desc[2])
ax.set_ylabel(iris_desc[3])
iris_centers = kmean_clustering(iris_data, 3, ax)
ax.scatter(iris_centers[:,2], iris_centers[:,3], s=50, c='r', label='centroid')
# plt.savefig('Lab6/iris_cluster_2.jpg', bbox='tight')

# %%
