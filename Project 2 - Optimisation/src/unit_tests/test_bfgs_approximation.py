import unittest
import numpy as np

from src.optimization_methods import BFGS
from scipy.optimize import fmin_bfgs

from src.optimization_problem import OptimizationProblem
from src.chebyquad_problem import chebyquad, gradchebyquad

class TestBfgsApproximation(unittest.TestCase):
	
	def setUp(self):
		self.rosen_brock = OptimizationProblem(lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
		
	def test_approximation_quality(self):
		for k in range(2, 7):
			x = np.linspace(0, 1, k)
			bfgs = BFGS(self.rosen_brock, k, initial_guess=x)

			bfgs.solve()
			H_k = bfgs.get_current_hessian_approximation()
	
			print(f"\nFor k={k}. BFGS hessian:\n{H_k}")
			
if __name__ == '__main__':
	unittest.main()