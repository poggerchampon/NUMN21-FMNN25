from .classical_newton_method import ClassicalNewtonMethod

from scipy.optimize import minimize_scalar
import numpy as np

# Using exact line search, returnn best step size alpha
# Overriding compute_alpha() in ClassicalNewtonMethod class
class NewtonExactLineSearch(ClassicalNewtonMethod):
	def _compute_alpha(self, x, direction):
		# define g(alpha) = f(x + alpha * direction)
		def g(alpha):
			return self.opt_problem.evaluate(x + alpha * direction)
		
		# use scipy to minimize the function
		result = minimize_scalar(g, bracket=[0, 1])
		
		if result.success:
			return result.x # return best alpha
		else:
			raise ValueError("Line search failed: " + result.message)
