from helper_methods import numerical_hessian
from helper_methods import numerical_gradient
from helper_methods import inv_numerical_hessian

# Super class for optimization method
class OptimizationMethod:
	def __init__(self, opt_problem):
		self.opt_problem = opt_problem
		
	def solve(self):
		raise NotImplementedError("The solve method should be implemented by derived classes")
	
	
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

		number_input_parameters=len(self.inspect.signature(self.objective_func))
		
		starting_point=np.zeros(number_input_parameters)

		

		#for _ in range(0,max_iterations):
		
			


		print(f"Solving using Quasi-Newton Method X with params {self.param1}, {self.param2}")
		# lots of math...

class NewtonExactLineSeach(OptimizationMethod):
	def __init__(self, opt_problem, h=1e-5, tolerance=1e-5):
		super().__init__(opt_problem)
		self.h = h
		self.tolerance = tolerance
		
	def solve(self):
		starting_point = self.opt_problem.get_num_of_paramaters()
		iteration = 0
		
		while True:
			current_value = self.opt_problem.evaluate(x)
			current_gradient = self.opt_problem.gradient(x)
			
			# Stopping criteria
			if np.linalg.norm(current_gradient) < self.tolerance:
				break
