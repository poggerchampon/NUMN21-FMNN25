import numpy as np
import scipy.sparse as sp
#from scipy.linalg import solve

def _initialise_equation_system(u, n, m, dx):
	"""
	Initialises the matrix 'A' and 'b' for the equation system 'Ax = b'
	Uses 2nd order central differences

	Parameters:
	------------
	'u' : A 2D numpy array representing the room including its boundaries
	'n' : Width of room 'u'
	'm' : Height of room 'u'
	'dx' : grid mesh width

	Returns:
	-----------
	'A' : A 2D numpy array representing the cofficients of the unkowns
	'b' : A 1D numpy array representing the constants
	"""
	
	data = []
	diags = []
	
	main_diag = np.zeros(n * m)
	left_diag = np.zeros(n * m - 1)
	right_diag = np.zeros(n * m - 1)
	upper_diag = np.zeros(n * m - m)
	lower_diag = np.zeros(n * m - m)
	
	b = np.zeros(n * m)
	# Will change this ugly creation of b, A works now at least
	for i in range(n):
		for j in range(m):
			idx = i * m + j
			if i == 0 or i == n - 1 or j == 0 or j == m - 1:
				main_diag[idx] = 1 / dx**2
				b[idx] = u[i, j] / dx**2
			else:
				main_diag[idx] = -4 / dx**2
				left_diag[idx - 1] = 1 / dx**2
				right_diag[idx] = 1 / dx**2
				upper_diag[idx] = 1 / dx**2 if idx + m < n * m else 0
				lower_diag[idx - m] = 1 / dx**2
				
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
	
	A, b = initialise_equation_system_sparse(u, n, m, dx)
	u_new = sp.linalg.spsolve(A, b).reshape((n, m))
	return u_new
