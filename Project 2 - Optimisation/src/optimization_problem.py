import matplotlib.pyplot as plt
import numpy as np
import importlib.util
import inspect

try:
	import tensorflow as tf
	TF_INSTALLED = True
except ImportError:
	TF_INSTALLED = False

from src.functions import numerical_gradient

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
	def gradient(self, x, method='tf'):
		if self.gradient_func:
			return self.gradient_func(x)
		else:
			# Use tensorflow if installed
			if method == 'tf' and TF_INSTALLED:
				x = tf.convert_to_tensor(x, dtype=tf.float32)
				with tf.GradientTape() as tape:
					tape.watch(x)
					y = self.objective_func(x)
				grads = tape.gradient(y, x)
				return grads.numpy()
			# Use central numerical differentiation
			elif method == 'numerical':
				return numerical_gradient(self.evaluate, x)
			# Invalid method choice
			else:
				raise ValueError("Invalid method. Use either 'tf' or 'numerical'")

	# Returns the function of the optimisation problem
	def get_function(self):
		return self.objective_func
	
	# Returns the evaluate function
	def get_evaluate(self):
		return self.evaluate
	
	def get_gradient(self):
		return self.gradient
	
	# Returns the num of paramaters for the function
	def get_num_of_parameters(self):
		return len(inspect.signature(self.objective_func).parameters)
