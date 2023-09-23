from .optimization_method import OptimizationMethod
from src.functions import approximate_hessian

from tqdm import tqdm

import numpy as np

# Super class for Newton Line search where line_search()
# is implemented by subclasses
class NewtonLineSearch(OptimizationMethod):
	def __init__(self, opt_problem, n, h=1e-12, tolerance=1e-5, max_iterations=1000, initial_guess=None):
		super().__init__(opt_problem)
		self.n = n
		self.h = h
		self.tolerance = tolerance
		self.max_iterations = max_iterations
		self.initial_guess = initial_guess
		self.path = [] # Initiate an empty list to store the optimisation path
		
		# Check paramaters
		self.validate_params()
		
	def line_search(self):
		raise NotImplementedError("This method should be implemented by subclass")
		
	def solve(self):	
		evaluate_func = self.opt_problem.get_evaluate()
		gradient_func = self.opt_problem.get_gradient()
		
		# Start with zeros if no initial guess is specified
		x = self.initial_guess if self.initial_guess is not None else np.zeros(self.n)
		iteration = 0
		
		# Use tqdm for neat progress bar
		with tqdm(total=self.max_iterations, desc="Optimizing", unit="iteration") as pbar:
			while True:
				current_value = evaluate_func(x)
				current_gradient = gradient_func(x)
				
				# Stopping criteria
				if np.linalg.norm(current_gradient) < self.tolerance:
					return x
				
				if iteration >= self.max_iterations:
					return x
				
				H = approximate_hessian(evaluate_func, gradient_func, x, self.n)
				G = 0.5 * (H + H.T) # Make symmetric if necessary
				direction = -np.linalg.solve(G, current_gradient) # Newton's direction
				
				# Get the best alpha using line search
				alpha = self.line_search(x, direction)
				
				# Update the current point and save it
				x += alpha * direction
				self.path.append(x.copy())
				
				# Update the progress bar
				pbar.update(1)
				iteration += 1
				
			# Minimum found
			return x
		
	def validate_params(self):
		# Check that parameters are greater than zero
		if self.h <= 0 or self.tolerance <= 0 or self.max_iterations <= 0 or self.n <= 0:
			raise ValueError("Parameters should be greater than zero")
		
		if self.initial_guess is not None and len(self.initial_guess) != self.n:
			raise ValueError("Initial guess must have same dimension as input function")
