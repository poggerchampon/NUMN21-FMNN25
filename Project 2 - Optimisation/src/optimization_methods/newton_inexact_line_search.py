from .classical_newton_method import ClassicalNewtonMethod
import numpy as np

class NewtonInexactLineSearch(ClassicalNewtonMethod):
	def _compute_alpha(self, x, direction, sigma=1e-4, max_backtracks=10, rho = 0.9):
		alpha = 1
		
		f_x = self.opt_problem.evaluate(x)
		gradient_f_x = self.opt_problem.gradient(x)
		
		for _ in range(max_backtracks):
			# Compute Armijo condition
			f_new_x = self.opt_problem.evaluate(x + alpha * direction)
			armijo_condition = f_x + sigma * alpha * np.dot(gradient_f_x, direction)
			
			# Check Armijo rule
			if f_new_x >= armijo_condition:
				alpha *= 0.5
			else:
				# Check Wolfe's condition
				gradient_new_f_x = self.opt_problem.gradient(x + alpha * direction)
				wolfe_condition = rho * np.dot(gradient_f_x, direction)
				if np.dot(gradient_new_f_x, direction) < wolfe_condition:
					alpha *= 2
				else:
					return alpha
		
		return alpha # Return the latest alpha, even though it didn't satisfy the condition
