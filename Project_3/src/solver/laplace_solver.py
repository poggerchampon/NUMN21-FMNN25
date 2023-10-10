import numpy as np
from scipy.linalg import solve

def _initialise_equation_system(u, n, m):
	"""
	Initialises the matrix 'A' and 'b' for the equation system 'Ax = b'
	Uses 2nd order central differences

	Parameters:
	------------
	'u' : A 2D numpy array representing the room including its boundaries
	'n' : Width of room 'u'
	'm' : Height of room 'u'

	Returns:
	-----------
	'A' : A 2D numpy array representing the cofficients of the unkowns
	'b' : A 1D numpy array representing the constants
	"""
	A = np.zeros((n * m, n * m))
	b = np.zeros(n * m)
	
	for i in range(n):
		for j in range(m):
			idx = i * m + j
			if i == 0 or i == n - 1 or j == 0 or j == m - 1:
				A[idx, idx] = 1
				b[idx] = u[i, j] # boundary condition
			else:
				A[idx, idx] = -4
				A[idx, idx + 1] = 1
				A[idx, idx - 1] = 1
				A[idx, idx + m] = 1
				A[idx, idx - m] = 1
				b[idx] = 0
			
	return A, b

def solve_laplace(u):
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
	
	A, b = _initialise_equation_system(u, n, m)
				
	u_new = solve(A, b).reshape((n, m))
	return u_new
