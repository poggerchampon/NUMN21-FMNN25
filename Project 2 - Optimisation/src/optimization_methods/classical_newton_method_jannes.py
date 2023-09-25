from .optimization_method import OptimizationMethod
from src.functions import approximate_hessian, numerical_hessian, numerical_gradient, inv_approximate_hessian

import numpy as np

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
			invhessian=inv_approximate_hessian(self,point)
			gradient=numerical_gradient(self,point)
			point[i+1]=point[i]-invhessian.dot(gradient)
			if point[i+1]-point[i]<self.tolerance:
				return point[i]
			
		# probably no min found
			
		return point[-1]
	
