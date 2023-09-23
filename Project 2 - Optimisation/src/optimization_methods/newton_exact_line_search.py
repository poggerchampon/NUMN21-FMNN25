from .newton_line_search import NewtonLineSearch
from scipy.optimize import minimize_scalar

class NewtonExactLineSearch(NewtonLineSearch):
	def line_search(self, x, direction):
		# define g(alpha) = f(x + alpha * direction)
		def g(alpha):
			return self.opt_problem.evaluate(x + alpha * direction)
		
		# Cheat and use scipy to minimize the function
		result = minimize_scalar(g, bracket=[0, 1])
		
		if result.success:
			return result.x # return best alpha
		else:
			raise ValueError("Line search failed: " + result.message)
