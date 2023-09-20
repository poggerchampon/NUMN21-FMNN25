import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import importlib.util

def is_tensorflow_installed():
	tensorflow_spec = importlib.util.find_spec('tensorflow')
	return tensorflow_spec is not None

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
	def gradient(self, x):
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
			
	# Default numerical method if gradient is not specified
	def numerical_gradient(self, x, h=1e-5):
		return (self.evaluate(x + h) - self.evaluate(x - h)) / (2.0 * h)
	
	# Plot function with specified x_range and number of points
	def plot_objective_2D(self, range_x = None, num_points=100):
		if range_x is None:
			print("Range for x is not specified")
			return
		
		x_values = np.linspace(range_x[0], range_x[1], num_points)
		y_values = [self.evaluate(x) for x in x_values]
		
		plt.plot(x_values, y_values)
		plt.xlabel("x")
		plt.ylabel("Objective function value")
		plt.title("Objective function")
		plt.show()
	
	# 3D visualisation?
	def plot_objective_3D(self):
		return
		
	
