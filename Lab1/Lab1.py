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

D, U = np.linalg.eig(B)
eigvecs_t = U.T
print(f'D : {D}')
print(f'U : {U}')
print(f'U[1] . U[2] : {np.dot(U[:,0], U[:,1])}')
print(f'U x U.T : {U @ U.T}')

#%% 2. Random Numbers and Univariate Distribution
num_trial = 5
num_samples = [100, 1000, 5000]
num_bins = [10, 20, 30, 50]
bin_counts_var = np.array([], dtype = float)

uniform_random_fig, uniform_random_subfig = plt.subplots(len(num_samples), len(num_bins), sharex=True, figsize=(len(num_samples) * 4,len(num_bins) * 1))
for i in range(len(num_samples)):
    for j in range(len(num_bins)):
        x = np.random.rand(num_samples[i], 1)
        counts, bins, patches = uniform_random_subfig[i,j].hist(x, bins = num_bins[j])
        if(j == 0):
             uniform_random_subfig[i,j].set(ylabel = f'{num_samples[i]} samples')
        if(i == len(num_samples) - 1):
            uniform_random_subfig[i,j].set(xlabel = f'{num_bins[j]} bins')
        # var = np.var(counts/num_samples)
        # bin_counts_var = np.append(bin_counts_var, np.var(counts/num_samples))
plt.savefig('Lab1/uniform_random_numbers.png', bbox_inches='tight')

num_samples_guassian = [100, 500, 2000, 5000]
gaussian_random_fig, gaussian_random_subfig = plt.subplots(1, 4, figsize=(15,2))
for i in range(len(num_samples_guassian)):
    x = np.random.randn(num_samples_guassian[i], 1)
    counts, bins, patches = gaussian_random_subfig[i].hist(x, bins = num_bins[0])
    gaussian_random_subfig[i].set(xlabel = f'{num_samples_guassian[i]} samples')
    # var = np.var(counts/num_samples)
    # bin_counts_var = np.append(bin_counts_var, np.var(counts/num_samples))
plt.savefig('Lab1/gaussian_random_numbers.png', bbox_inches='tight')

sum_random_fig, sum_random_subfig = plt.subplots(1, 4, figsize=(15,2))
num_randoms = [2, 5, 10, 25]
num_samples_sum = [100, 500, 2000, 5000, 10000]
for k in range(len(num_randoms)):
    x = np.zeros(num_samples_sum[2])
    for i in range(num_samples_sum[2]): 
        x[i] = sum(np.random.rand(num_randoms[k], 1)) - sum(np.random.rand(num_randoms[k], 1))
    counts, bins, patches = sum_random_subfig[k].hist(x, num_bins[-1])
    sum_random_subfig[k].set(xlabel = f'sum of {num_randoms[k]} numbers')
plt.savefig('Lab1/sum_random_numbers.png', bbox_inches='tight')

#%% 3. Uncertainty in Estimation
max_trial = 2000
sample_size_range = np.linspace(100, 500, 100)
plot_var = np.zeros(len(sample_size_range))

uncer_fig, uncer_subfig = plt.subplots(1,2, figsize=(8,2))
for sample_size_index in range(len(sample_size_range)):
    num_samples = np.int(sample_size_range[sample_size_index])
    trail_vars = np.zeros(max_trial)
    for trial in range(max_trial):
        samples = np.random.randn(num_samples, 1) * 10
        trail_vars[trial] = np.var(samples)
    plot_var[sample_size_index] = np.var(trail_vars)
uncer_subfig[0].title.set_text('Estimated Variance of\nGaussian Sampling')
uncer_subfig[0].set_xlabel('#Samples')
uncer_subfig[0].set_ylabel('Variance')
uncer_subfig[0].plot(sample_size_range, plot_var)

plot_var = np.zeros(len(sample_size_range))
for sample_size_index in range(len(sample_size_range)):
    num_samples = np.int(sample_size_range[sample_size_index])
    trail_vars = np.zeros(max_trial)
    for trial in range(max_trial):
        samples = np.random.rand(num_samples, 1) * 10
        trail_vars[trial] = np.var(samples)
    plot_var[sample_size_index] = np.var(trail_vars)
uncer_subfig[1].title.set_text('Estimated Variance of\nUniform Sampling')
uncer_subfig[1].set_xlabel('#Samples')
uncer_subfig[1].set_ylabel('Variance')
uncer_subfig[1].plot(sample_size_range, plot_var)
plt.savefig('Lab1/estmd_variance.png', bbox_inches='tight')
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
C = np.array([[2,1], [1,2]], np.float32)
X1p, Y1p, Z1p = twoDGaussianPlot(nx, ny, m1, C)
m2 = np.array([2, 0])
C2 = np.array([[2,-1], [-1,2]], np.float32)
X2p, Y2p, Z2p = twoDGaussianPlot(nx, ny, m2, C2)
m3 = np.array([-2, -2])
C3 = np.array([[2,0], [0,2]], np.float32)
X3p, Y3p, Z3p = twoDGaussianPlot(nx, ny, m3, C3)
plt.contour(X1p, Y1p, Z1p, 3, colors = 'r')
plt.contour(X2p, Y2p, Z2p, 3, colors = 'g')
plt.contour(X3p, Y3p, Z3p, 3, colors = 'b')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('Contour plot of 3 bivariate Gaussian densities')
plt.savefig('Lab1/contours.png', bbox_inches='tight')

#%% 5. Sampling from a Multivariate Gaussian Distribution
C = [[2,1], [1,2]]
C2 = [[2,-1], [-1,2]]
A = np.linalg.cholesky(C)
A2 = np.linalg.cholesky(C2)

X = np.random.randn(10000, 2)
Y = X @ A
Y2 = X @ A2
plt.title('Scatter of Isotropic and Correlated Gaussian Densities')
plt.scatter(Y[:,0], Y[:,1], c='r', s=100, edgecolors='')
plt.scatter(X[:,0], X[:,1], c='g', s=100, edgecolors='')
plt.grid(True)
plt.savefig('Lab1/scatter_of_isotropic.png', bbox_inches='tight')

#%% 6. Distribution of Projections
theta = np.pi / 3
u = [np.cos(theta), np.sin(theta)]
y_proj = Y @ u
print('Dimension before projection : ', Y.shape)
print('Dimension after projection : ', y_proj.shape)
print('Original variance : ', np.var(Y))
print('Projected variance : ', np.var(y_proj))

num_theta = 360
y_proj_var = np.zeros(num_theta)
y2_proj_var = np.zeros(num_theta)
theta_range = np.linspace(0, 2 * np.pi, num_theta)
max_var = 0.0
max_u = [0.0, 0.0]
for i in range(num_theta):
    theta = theta_range[i]
    u = [np.cos(theta), np.sin(theta)]
    cur_var = np.var(Y @ u)
    y_proj_var[i] = np.var(Y @ u)
    y2_proj_var[i] = np.var(Y2 @ u)
plt.title('Variance of projections of Y according to theta')
plt.grid(True)
plt.xlabel('Theta')
plt.xlim(xmin=0, xmax=360)
plt.ylabel('Variance of Projection')
plt.plot(y_proj_var, c='r', label = 'C=[[ 2, 1],[ 1, 2]]')
plt.plot(y2_proj_var, c='g', label = 'C=[[ 2,-1],[-1, 2]]')
plt.legend()
plt.savefig('Lab1/variance_of_proj.png', bbox_inches='tight')

y_evals, y_evecs = np.linalg.eig([[2,1], [1,2]])
y_ev1 = y_evecs[:,0]
y_ev2 = y_evecs[:,1]
y2_evals, y2_evecs = np.linalg.eig([[2,-1], [-1,2]])
y2_ev1 = y2_evecs[:,0]
y2_ev2 = y2_evecs[:,1]
print('Y eigenvalues : ', y_evals)
print('Y eigenvector 1 : ', y_ev1)
print('Y eigenvector 2 : ', y_ev2)
print('Y2 eigenvalues : ', y2_evals)
print('Y2 eigenvector 1 : ', y2_ev1)
print('Y2 eigenvector 2 : ', y2_ev2)

print('Y1 Eigenvector-proj max variance : ', np.var(Y @ y_ev1))
print('Y1 Eigenvector-proj min variance : ', np.var(Y @ y_ev2))
print('Y2 Eigenvector-proj max variance : ', np.var(Y2 @ y2_ev1))
print('Y1 Eigenvector-proj min variance : ', np.var(Y @ y_ev1))

#%%
