from .optimization_method import OptimizationMethod
from src.functions import approximate_hessian

from scipy.optimize import minimize_scalar
from tqdm import tqdm

import numpy as np

# Can be specified to use either the exact line search
# or the inexact line search using Armijo rule
class NewtonLineSearch(OptimizationMethod):
	def __init__(self, opt_problem, n, h=1e-12, tolerance=1e-5, max_iterations=1000, initial_guess=None, exact=True):
		super().__init__(opt_problem)
		self.n = n
		self.h = h
		self.tolerance = tolerance
		self.max_iterations = max_iterations
		self.initial_guess = initial_guess
		self.exact = exact
		self.path = [] # Initiate an empty list to store the optimisation path
		
		# Check paramaters
		self.validate_params()
		
	def exact_line_search(self, x, direction):
		# define g(alpha) = f(x + alpha * direction)
		def g(alpha):
			return self.opt_problem.evaluate(x + alpha * direction)
		
		# Cheat and use scipy to minimize the function
		result = minimize_scalar(g, bracket=[0, 1])
		
		if result.success:
			return result.x # return best alpha
		else:
			raise ValueError("Line search failed: " + result.message)
			
	def inexact_line_search(self, x, direction, sigma=1e-4, max_backtracks=10):
		alpha = 1
		for _ in range(max_backtracks):
			# Compute Armijo condition
			f_new_x = self.opt_problem.evaluate(x + alpha * direction)
			f_x = self.opt_problem.evaluate(x)
			gradient_f_x = self.opt_problem.gradient(x)
			armijo_condition = f_x + sigma * alpha * np.dot(gradient_f_x, direction)
			
			# Check Armijo rule
			if f_new_x <= armijo_condition:
				return alpha
			alpha *= 0.5
			
	def solve(self):	
		evaluate_func = self.opt_problem.get_evaluate()
		gradient_fuc = self.opt_problem.get_gradient()
		
		# Start with zeros if no initial guess is specified
		x = self.initial_guess if self.initial_guess is not None else np.zeros(self.n)
		iteration = 0
		
		# Use tqdm for neat progress bar
		with tqdm(total=self.max_iterations, desc="Optimizing", unit="iteration") as pbar:
			while True:
				current_value = self.opt_problem.evaluate(x)
				current_gradient = self.opt_problem.gradient(x)
				
				# Stopping criteria
				if np.linalg.norm(current_gradient) < self.tolerance:
					return x
				
				if iteration >= self.max_iterations:
					return x
				
				H = approximate_hessian(evaluate_func, gradient_fuc, x, self.n)
				G = 0.5 * (H + H.T) # Make symmetric if necessary
				direction = -np.linalg.solve(G, current_gradient) # Newton's direction
				
				# Choose between exact and inexact line search
				if self.exact:
					alpha = self.exact_line_search(x, direction)
				else:
					alpha = self.inexact_line_search(x, direction)
				
				# Update the current point and save it
				x += alpha * direction
				self.path.append(x.copy())
				
				# Update the progress bar
				pbar.update(1)
				iteration += 1
				
			# Minimum found
			return x
		
	# Check that parameters are greater than zero
	def validate_params(self):
		# Check that parameters are greater than zero
		if self.h <= 0 or self.tolerance <= 0 or self.max_iterations <= 0 or self.n <= 0:
			raise ValueError("Parameters should be greater than zero")
		
		if self.initial_guess is not None and len(self.initial_guess) != self.n:
			raise ValueError("Initial guess must have same dimension as input function")
