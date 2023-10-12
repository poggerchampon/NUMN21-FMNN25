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
	main_diag = np.zeros(n * m)
	left_diag = np.zeros(n * m - 1)
	right_diag = np.zeros(n * m - 1)
	upper_diag = np.zeros(n * m - m)
	lower_diag = np.zeros(n * m - m)
	
	b = np.zeros(n * m)
	
	# Indices for boundary conditions
	boundary_indices = np.concatenate([
		np.arange(0, m),  # Top boundary
		np.arange((n-1) * m, n * m),  # Bottom boundary
		np.arange(0, n * m, m),  # Left boundary
		np.arange(m-1, n * m, m)  # Right boundary
	])
	
	# Boundary conditions
	main_diag[boundary_indices] = 1 / dx**2
	b[boundary_indices] = u.flatten()[boundary_indices] / dx**2
	
	# Mask for interior points, we don't want to overwrite the boundary conditions
	interior_mask = np.ones((n, m), dtype=bool)
	interior_mask[np.unravel_index(boundary_indices, (n, m))] = 0
	
	# Interior indices
	interior_indices = np.arange(n * m).reshape(n, m)[interior_mask].flatten()
	
	# Set up A interior
	main_diag[interior_indices] = -4 / dx**2
	left_diag[interior_indices - 1] = 1 / dx**2
	right_diag[interior_indices] = 1 / dx**2
	upper_diag[interior_indices] = 1 / dx**2
	lower_diag[interior_indices - m] = 1 / dx**2
				
	data = [lower_diag, left_diag, main_diag, right_diag, upper_diag]
	diags = [-m, -1, 0, 1, m]
	
	A = sp.diags(data, diags, shape=(n * m, n * m), format='csr')
	
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
