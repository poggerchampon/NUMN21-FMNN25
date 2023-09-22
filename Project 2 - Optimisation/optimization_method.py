from helper_methods import numerical_hessian
from helper_methods import numerical_gradient
from helper_methods import inv_numerical_hessian
from helper_methods import approximate_hessian

from scipy.optimize import minimize_scalar
from tqdm import tqdm

import inspect
import numpy as np

# Super class for optimization method
class OptimizationMethod:
	def __init__(self, opt_problem):
		self.opt_problem = opt_problem
		
	def solve(self):
		raise NotImplementedError("The solve method should be implemented by derived classes")

	# Method to validate paramaters
	def validate_params(self):
		pass # implemented by derived classes
	
#------------- Derived Quasi Newton Methods -----------------------
	
class ClassicalNewtonMethod(OptimizationMethod):
	def __init__(self, opt_problem, h=1e-5, tolerance=1e-5,max_iterations=1000):
		# initialise method
		super().__init__(opt_problem)
		
		# initialise paramaters
		self.h = h
		self.tolerance = tolerance
		self.max_iterations=max_iterations

		if self.h <= 0:
			raise ValueError("Invalid parameter. h should be bigger than 0")
		if self.tolerance <= 0:
			raise ValueError("Invalid parameter. tolerance should be bigger than 0")
		if self.max_iterations <= 0:
			raise ValueError("Invalid parameter. max_iterations should be bigger than 0")

	def solve(self):

		import numpy as np
		
		number_input_parameters=self.opt_problem.get_num_of_parameters()
		
		point=np.zeros(number_input_parameters)		#starting at 0
		
		print(f"Solving using Quasi-Newton Method X with params {self.param1}, {self.param2}")

		for i in range(0,self.max_iterations-1):
			invhessian=inv_numerical_hessian(self,point)
			gradient=numerical_gradient(self,point)
			point[i+1]=point[i]-invhessian.dot(gradient)
			if point[i+1]-point[i]<self.tolerance:
				return point[i]

		# probably no min found

		return point[-1]

class NewtonExactLineSearch(OptimizationMethod):
	def __init__(self, opt_problem, n, h=1e-12, tolerance=1e-5, max_iterations=1000):
		super().__init__(opt_problem)
		self.n = n
		self.h = h
		self.tolerance = tolerance
		self.max_iterations = max_iterations
		
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
		
		# Start with zeroscan
		x = np.random.random(self.n)
		iteration = 0
		
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
			
				# Update the current point
				x += alpha * direction
			
				iteration += 1
			
				# Update the progress bar
				pbar.update(1)
		
			# Minimum found
			return x
			
	# Check that parameters are greater than zero
	def validate_params(self):
		if self.h <= 0 or self.tolerance <= 0:
			raise ValueError("Paramaters should be greater than zero")
