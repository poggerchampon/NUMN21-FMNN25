import numpy as np
import scipy.sparse as sp

def _initialise_equation_system_sparse(u, n, m, dx):
	"""
	Initialises the sparse matrix 'A' and 'b' for the equation system 'Ax = b'
	Uses 2nd order central differences

	Parameters:
	------------
	'u' : A 2D numpy array representing the room including its boundaries
	'n' : Width of room 'u'
	'm' : Height of room 'u'
	'dx' : grid mesh width

	Returns:
	-----------
	'A' : A 2D sparse matrix representing the coefficients of the unknowns
	'b' : A 1D numpy array representing the constants
	"""
	N = n * m
	coeff = 1 / dx**2
	
	main_diag = np.zeros(N)
	left_diag, right_diag = np.zeros(N - 1), np.zeros(N - 1)
	upper_diag, lower_diag = np.zeros(N - m), np.zeros(N - m)
	
	b = np.zeros(N)
	
	# Boundary indices
	top, bottom = np.arange(m), np.arange((n-1)*m, N)
	left, right = np.arange(0, N, m), np.arange(m-1, N, m)
	boundary_indices = np.concatenate([top, bottom, left, right])
	
	main_diag[boundary_indices] = coeff
	b[boundary_indices] = u.flatten()[boundary_indices] * coeff
	
	# Mask for interior points, we don't want to overwrite the boundary conditions
	interior_mask = np.ones((n, m), dtype=bool)
	interior_mask[np.unravel_index(boundary_indices, (n, m))] = 0
	interior_indices = np.arange(N).reshape(n, m)[interior_mask].flatten()
	
	# Set up A interior
	main_diag[interior_indices] = -4 * coeff
	left_diag[interior_indices - 1] = coeff
	right_diag[interior_indices] = coeff
	upper_diag[interior_indices] = coeff
	lower_diag[interior_indices - m] = coeff
				
	data = [lower_diag, left_diag, main_diag, right_diag, upper_diag]
	diags = [-m, -1, 0, 1, m]
	
	A = sp.diags(data, diags, shape=(N, N), format='csr')
	
	return A, b

def solve_laplace(u, dx):
	"""
	Solves the laplace equation for a room 'u' using Scipy.linalg.solve
	
	Parameters:
	------------
	'u' : A 2D numpy array representing the room including its boundaries

	Returns:
	-----------
	'u_new' : A 2D numpy array representing the solution, room with updated temperatures
	"""
	n = u.shape[0]
	m = u.shape[1]
	
	A, b = _initialise_equation_system_sparse(u, n, m, dx)
	u_new = sp.linalg.spsolve(A, b).reshape((n, m))
	
	return u_new
