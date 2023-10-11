import numpy as np
from scipy.interpolate import interp1d

def interpolate_boundary(u_received, new_length):
	"""
	Interpolate the given data to a new length using linear interpolation.
	
	Parameters:
	- u_received (numpy array): The data to be interpolated.
	- new_length (int): The desired length of the interpolated data.
	
	Returns:
	- numpy array: Interpolated data of the specified new length.
	"""
	
	old_length = len(u_received)
	x_old = np.linspace(0, 1, old_length)
	x_new = np.linspace(0, 1, new_length)
	f = interp1d(x_old, u_received, kind='linear')
	return f(x_new)
