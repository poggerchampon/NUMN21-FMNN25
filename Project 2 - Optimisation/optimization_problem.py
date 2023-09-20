import matplotlib.pyplot as plt
import numpy as np
import importlib.util

def is_tensorflow_installed():
	try:
		import tensorflow as tf
		return True
	except ImportError:
		return False

class OptimizationProblem:
	# Initiate an objective function with option to specify gradient
	def __init__(self, objective_func, gradient_func=None):
		if not callable(objective_func):
			raise TypeError("Specified objective_func must be callable")
		if gradient_func is not None and not callable(gradient_func):
			raise TypeError("Specified gradient_func must be callable")
		
		self.objective_func = objective_func
		self.gradient_func = gradient_func
		
	# Evaluates itself and returns the function output
	def evaluate(self, x):
		return self.objective_func(x)
	
	# Returns the gradient value of the function
	def gradient(self, x, method='numerical'):
		if self.gradient_func:
			return self.gradient_func(x)
		else:
			# Use tensorflow if installed
			if method == 'tf' and is_tensorflow_installed():
				x = tf.convert_to_tensor(x, dtype=tf.float32)
				with tf.GradientTape() as tape:
					tape.watch(x)
					y = self.objective_func(x)
				grads = tape.gradient(y, x)
				return grads.numpy()
			# Use central numerical differentiation
			elif method == 'numerical':
				return numerical_gradient(x)
			# Invalid method choice
			else:
				raise ValueError("Invalid method. Use either 'tf' or 'numerical'")

	# Returns the function of the optimisation problem
	def get_function(self):
		return self.objective_func
	
	# Returns the num of paramaters for the function
	def get_num_of_parameters(self):
		return len(self.inspect.signature(self.objective_func))
		
# ------------------- Optional additions ------------------------
			
# moved to helper methods	
