


# Super class for optimization method
class OptimizationMethod:
	def __init__(self, opt_problem):
		self.opt_problem = opt_problem
		
	def solve(self):
		raise NotImplementedError("The solve method should be implemented by derived classes")
	
	
#------------- Derived Quasi Newton Methods -----------------------
	
class ClassicalNewtonMethod(OptimizationMethod):
	def __init__(self, opt_problem, h=1e-5, tolerance=1e-5):
		# initialise method
		super().__init__(opt_problem)
		
		# initialise paramaters
		self.h = h
		self.tolerance = tolerance
		
		
	def solve(self):

		import numpy as np

		number_input_parameters=len(self.inspect.signature(self.objective_func))
		max_iterations=1000
		starting_point=np.zeros(number_input_parameters)

		

		#for _ in range(0,max_iterations):

			


		print(f"Solving using Quasi-Newton Method X with params {self.param1}, {self.param2}")
		# lots of math...