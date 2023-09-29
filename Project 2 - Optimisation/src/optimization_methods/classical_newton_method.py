from .optimization_method import OptimizationMethod
from src.functions import approximate_hessian

from tqdm import tqdm
import numpy as np

class ClassicalNewtonMethod(OptimizationMethod):
	def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-5, max_iterations=1000, initial_guess=None):
		super().__init__(opt_problem)
		self.n = n
		self.h = h
		self.tolerance = tolerance
		self.max_iterations = max_iterations
		self.initial_guess = initial_guess
		
		# Store previous x and gradient
		self.prev_x = None
		self.prev_gradient = None
		self.path = []  # Initiate an empty list to store the optimisation path
		
		# Check parameters
		self.validate_params()
		
	# Method for finding step size alpha. 
	# Is overridden in derived line search methods
	def compute_alpha(self, x, direction):
		# In classical Newton, alpha is always 1
		return 1.0
	
	# Method for finding the direction. Default here is classical newton
	# Can be overridden in derived classes like DFP, SimpleBroyden, BFGS etc
	def compute_direction(self, x, gradient_func, current_gradient):
		H = approximate_hessian(gradient_func, x, self.n)
		G = 0.5 * (H + H.T)  # Make symmetric if necessary
		
		direction = -np.linalg.solve(G, current_gradient)  # Newton's direction
		return direction
		
	# hessian update method for Quasi newton methods
	def update_inv_hessian(self, x, gradient_func, current_gradient):
		pass
	
	def solve(self):	
		gradient_func = self.opt_problem.get_gradient()
		
		# Start with zeros if no initial guess is specified
		x = self.initial_guess if self.initial_guess is not None else np.zeros(self.n)
		iteration = 0
		
		# Use tqdm for neat progress bar
		with tqdm(total=self.max_iterations, desc="Optimizing", unit="iteration") as pbar:
			while True:
				current_gradient = gradient_func(x)
				
				x_str = np.array2string(x, formatter={'all': lambda x: f'{x:.2e}'})
				pbar.set_description(f"Optimizing (x = {x_str})")
				
				# Stopping criteria
				if np.linalg.norm(current_gradient) < self.tolerance or iteration >= self.max_iterations:
					return x
				
				# Get direction and step size using either classical method or derived method 
				direction = self.compute_direction(x, gradient_func, current_gradient)
				alpha = self.compute_alpha(x, direction)
				
				# update hessian
				self.update_inv_hessian(x, gradient_func, current_gradient)
				
				self.prev_x = x.copy()
				self.prev_gradient = current_gradient
				
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
			
