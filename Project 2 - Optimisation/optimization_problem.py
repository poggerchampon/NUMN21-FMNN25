import matplotlib.pyplot as plt
import numpy as np

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
			# Option 1. Numerical gradient computation? (fallback)
			# Option 2. Raise "NotImplementedError" (boring)
			# Option 3. Use Pytorch or Tensorflow to compute gradients 
			raise NotImplementedError("Gradient function is not specified")
		
		
		
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
		
	
