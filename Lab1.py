#%% COMP6245 - Lab 1

#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#%% 1. Basic Linear Algebra
x = np.array([1,2])
y = np.array([-2,1])
x_dot_y = np.dot(x, y)
print(f'x.y : {x_dot_y}')

norm_x = np.linalg.norm(x)
norm_y = np.linalg.norm(y)
c = np.sqrt(x[0]**2 + x[1]**2)
print(f'|x| : {norm_x}')

theta = np.arccos(np.dot(x,y) / (norm_x * norm_y))
print(f'Theta : {theta * 180 / np.pi}')

B = np.array([[3,2,1], [2,6,5], [1,5,9]], dtype = float)
print(f'B : {B}')
print(f'B - B.T : {B - B.T}') # B is a symmetric matrix (For all elements in B, B[i,j] = B[j,i]). That's why B - B.T is a zeros matrix.

Z = np.random.rand(3)
v = B @ Z
print(f'Z : {Z}')
print(f'Dimension of B x Z : {v.shape}')
print(f'B x Z : {v}') # B has a dimension of 3 x 3, while Z has a dimension of 3 x 1. Therefore, the dimension of B x Z = 3 x 1
print(f'Z.T x B x Z : {Z.T @ B @ Z}')

trace_B = np.trace(B)
det_B = np.linalg.det(B)
inv_B = np.linalg.inv(B)
print(f'tr(B) : {trace_B}')
print(f'det(B) : {det_B}')
print(f'inv(B) : {inv_B}')
print(f'B x B(-1) : {B @ inv_B}')
print(f'Is B x B(-1) an identity matrix : {np.allclose(B @ inv_B, np.eye(B.shape[0]))}')

eigvals, eigvecs = np.linalg.eig(B)
eigvecs_t = eigvecs.T
print(f'Eigenvalues of B : {eigvals}')
print(f'Eigenvectors of B : {eigvecs}')

#%% 2. Random Numbers and Univariate Distribution
num_samples = 1000
num_bins = 20
bin_counts_var = np.array([], dtype = float)

for trial in range(10):
    x = np.random.rand(num_samples, 1)
    plt.figure(figsize=(3, 3,))
    plt.title(label = f'Trial #{trial}')
    plt.xlabel('Bin')
    plt.ylabel('Counts')
    counts, bins, patches = plt.hist(x, bins = num_bins)
    var = np.var(counts/num_samples)
    bin_counts_var = np.append(bin_counts_var, np.var(counts/num_samples))

for trial in range(10):
    x = np.zeros(num_samples)
    for i in range(num_samples): x[i] = sum(np.random.rand(10, 1)) + sum(np.random.rand(10, 1))
    plt.figure(figsize=(3, 3,))
    plt.title(label = f'Trial #{trial}')
    plt.xlabel('Bin')
    plt.ylabel('Counts')
    plt.hist(x, num_bins, color='b', alpha=0.8, rwidth=0.8)

#%% 3. Uncertainty in Estimation
max_trial = 2000
sample_size_range = np.linspace(100, 500, 40)
plot_var = np.zeros(len(sample_size_range))
for sample_size_index in range(len(sample_size_range)):
    num_samples = np.int(sample_size_range[sample_size_index])
    trail_vars = np.zeros(max_trial)
    for trial in range(max_trial):
        samples = np.random.randn(num_samples, 1)
        trail_vars[trial] = np.var(samples)
    plot_var[sample_size_index] = np.var(trail_vars)
plt.title('Variance in Estimating Variance')
plt.xlabel('#Samples')
plt.ylabel('Variance')
plt.plot(sample_size_range, plot_var)

#%% 4. Bivariate Gaussian Distribution
def gauss2D(x, m, C):
    Ci = np.linalg.inv(C)
    dC = np.linalg.det(C)
    num = np.exp(-0.5 * np.dot((x -  m).T, np.dot(Ci, (x-m))))
    den = 2 * np.pi * dC
    return num / den

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

nx, ny = 50, 40
m1 = np.array([0, 2])
C1 = np.array([[2,1], [1,2]], np.float32)
X1p, Y1p, Z1p = twoDGaussianPlot(nx, ny, m1, C1)
m2 = np.array([2, 0])
C2 = np.array([[2,-1], [-1,2]], np.float32)
X2p, Y2p, Z2p = twoDGaussianPlot(nx, ny, m2, C2)
m3 = np.array([-2, -2])
C3 = np.array([[2,0], [0,2]], np.float32)
X3p, Y3p, Z3p = twoDGaussianPlot(nx, ny, m3, C3)
plt.contour(X1p, Y1p, Z1p, 3)
plt.contour(X2p, Y2p, Z2p, 3)
plt.contour(X3p, Y3p, Z3p, 3)
plt.grid(True)

#%%
