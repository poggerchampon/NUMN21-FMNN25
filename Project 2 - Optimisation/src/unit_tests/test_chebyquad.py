import unittest
import numpy as np

from src.optimization_methods import BFGS
from scipy.optimize import fmin_bfgs

from src.optimization_problem import OptimizationProblem
from src.chebyquad_problem import chebyquad, gradchebyquad

class TestChebyquad(unittest.TestCase):
	
	def setUp(self):
		self.chebyquad = OptimizationProblem(chebyquad, gradchebyquad)
		
	def approximation_error(self, result1, result2):
		return np.linalg.norm(result1 - result2)
			
	def test_compare_BFGS_n_4(self):
		print("\nTesting BFGS and scipy.optimize.fmin_bfgs with n = 4")
		initial_guess = np.linspace(0, 1, 4)
		
		bfgs_solver = BFGS(self.chebyquad, 4, initial_guess=initial_guess, progress_bar=False)
		bfgs_result = bfgs_solver.solve()
		scipy_result = fmin_bfgs(self.chebyquad.get_evaluate(), initial_guess, fprime=self.chebyquad.get_gradient())
		
		print(f"With n: 4, custom bfgs gave after {len(bfgs_solver.path)} iterations the result:\n {bfgs_result}. \nScipy:\n {scipy_result}")
		print(f"Error: {self.approximation_error(bfgs_result, scipy_result)}")
		
		np.testing.assert_allclose(bfgs_result, scipy_result, atol=1e-3)
	
	def test_compare_BFGS_n_7(self):
		print("\nTesting BFGS and scipy.optimize.fmin_bfgs with n = 7")
		initial_guess = np.linspace(0, 1, 7)
		
		bfgs_solver = BFGS(self.chebyquad, 7, initial_guess=initial_guess, progress_bar=False)
		bfgs_result = bfgs_solver.solve()
		scipy_result = fmin_bfgs(self.chebyquad.get_evaluate(), initial_guess, fprime=self.chebyquad.get_gradient())
		
		print(f"With n: 7, custom bfgs gave after {len(bfgs_solver.path)} iterations the result:\n {bfgs_result}. \nScipy:\n {scipy_result}")
		print(f"Error: {self.approximation_error(bfgs_result, scipy_result)}")
		
		np.testing.assert_allclose(bfgs_result, scipy_result, atol=1e-3)
		
	def test_compare_BFGS_n_11(self):
		print("\nTesting BFGS and scipy.optimize.fmin_bfgs with n = 11")
		initial_guess = np.linspace(0, 1, 11)
		
		bfgs_solver = BFGS(self.chebyquad, 11, initial_guess=initial_guess, progress_bar=False)
		bfgs_result = bfgs_solver.solve()
		scipy_result = fmin_bfgs(self.chebyquad.get_evaluate(), initial_guess, fprime=self.chebyquad.get_gradient())
		
		print(f"With n: 11, custom bfgs gave after {len(bfgs_solver.path)} iterations the result:\n {bfgs_result}. \nScipy:\n {scipy_result}")
		print(f"Error: {self.approximation_error(bfgs_result, scipy_result)}")
		
		np.testing.assert_allclose(bfgs_result, scipy_result, atol=1e-3)
		
# Run all unit tests
if __name__ == "__main__":
	unittest.main()