import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.optimization_methods import OptimizationMethod, NewtonInexactLineSearch
from src.optimization_problem import OptimizationProblem
from src.functions import save_rosenbrock_plot

class TestNewtonInexactLineSearch(unittest.TestCase):
	def setUp(self):
		self.rosen_brock = OptimizationProblem(lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
	
	def test_rosen_brock(self):
		print("\nTesting Rosenbrock")
		solver = NewtonInexactLineSearch(self.rosen_brock, n=2)
		expected_solution = np.array([1.0, 1.0])
		
		# Assert check -> is result close enough to expected
		np.testing.assert_allclose(solver.solve(), expected_solution, atol=1e-1)
		
		# Save plot of the optimisation path
		save_rosenbrock_plot(np.array(solver.path))
		
# Run all unit tests
if __name__ == "__main__":
	unittest.main()
