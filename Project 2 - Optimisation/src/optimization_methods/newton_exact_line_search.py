from .optimization_method import OptimizationMethod
from src.functions.helper_methods import approximate_hessian

from scipy.optimize import minimize_scalar
from tqdm import tqdm

import numpy as np

class NewtonExactLineSearch(OptimizationMethod):
	def __init__(self, opt_problem, n, h=1e-12, tolerance=1e-5, max_iterations=1000):
		super().__init__(opt_problem)
		self.n = n
		self.h = h
		self.tolerance = tolerance
		self.max_iterations = max_iterations
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
			
	def solve(self):	
		evaluate_func = self.opt_problem.get_evaluate()
		gradient_fuc = self.opt_problem.get_gradient()
		
		# Start with zeros
		x = np.random.random(self.n)
		iteration = 0
		
		# Use tqdm for neat progress bar
		with tqdm(total=self.max_iterations, desc="Optimizing", unit="iteration") as pbar:
			while True:
				current_value = self.opt_problem.evaluate(x)
				current_gradient = self.opt_problem.gradient(x)
				
				# Stopping criteria
				if np.linalg.norm(current_gradient) < self.tolerance:
					print(f"Converged in {iteration} iterations")
					return x
				
				if iteration >= self.max_iterations:
					print("Maximum iterations reached.")
					return x
				
				# Compute approximate Hessian
				H = approximate_hessian(evaluate_func, gradient_fuc, x, self.n)
				
				# Make symmetric if necessary
				G = 0.5 * (H + H.T)
				
				# Newton's direction
				direction = -np.linalg.solve(G, current_gradient)
				
				# Do exact line search to find the optimal step size
				alpha = self.exact_line_search(x, direction)
				
				# Update the current point and record it
				x += alpha * direction
				self.path.append(x.copy())
				
				iteration += 1
				
				# Update the progress bar
				pbar.update(1)
				
			# Minimum found
			return x
		
	# Check that parameters are greater than zero
	def validate_params(self):
		if self.h <= 0 or self.tolerance <= 0:
			raise ValueError("Paramaters should be greater than zero")
