#%% COMP6245 - Lab 1

#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#%% Basic Linear Algebra
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

E_VALS, E_VECS = np.linalg.eig(B)
print(f'Eigenvalues of B : {E_VALS}')
print(f'Eigenvectors of B : {E_VECS}')
print(f'Eigenvector[1] . Eigenvector[2] : {np.fix(np.dot(E_VECS[:,0], E_VECS[:,1]))}')
print(f'Eigenvector[1] . Eigenvector[3] : {np.fix(np.dot(E_VECS[:,0], E_VECS[:,2]))}')
print(f'Eigenvector[2] . Eigenvector[3] : {np.fix(np.dot(E_VECS[:,1], E_VECS[:,2]))}')
print(f'E_VECS @ E_VECS.T : {E_VECS @ E_VECS.T}')