import unittest
import numpy as np

from src.optimization_methods import NewtonExactLineSearch
from src.optimization_problem import OptimizationProblem
from src.functions import save_rosenbrock_plot

class TestNewtonExactLineSearch(unittest.TestCase):
	
	def setUp(self):
		self.mock_problem = OptimizationProblem(lambda x: np.dot(x, x), lambda x: 2 * x)
		self.rosen_brock = OptimizationProblem(lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
		
	def test_validate_params_negative(self):
		print("\nTesting validate_params_negative")
		with self.assertRaises(ValueError):
			opt_method = NewtonExactLineSearch(self.mock_problem, n=2, h=-1, tolerance=-1)
			
	def test_max_iterations_stopping_criteria(self):
		print("\nTesting max_iterations stop")
		opt_method = NewtonExactLineSearch(self.mock_problem, n=2, max_iterations=1)
		result = opt_method.solve()
		
		# For a mock quadratic function and only 1 iteration, the result should be [0, 0]
		np.testing.assert_array_almost_equal(result, np.array([0, 0]))
			
	def test_rosen_brock(self):
		print("\nTesting Rosenbrock")
		solver = NewtonExactLineSearch(self.rosen_brock, n=2)
		expected_solution = np.array([1.0, 1.0])
		
		# Check if result is close enough to expected
		np.testing.assert_allclose(solver.solve(), expected_solution, atol=1e-1)
		
		# Save plot of optimisation path
		save_rosenbrock_plot(np.array(solver.path), filename='exact_search_contour')
		
# Run all unit tests
if __name__ == "__main__":
	unittest.main()
