import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.optimization_methods import OptimizationMethod, NewtonInexactLineSearch
from src.optimization_problem import OptimizationProblem

class TestNewtonInexactLineSearch(unittest.TestCase):
	def setUp(self):
		self.rosen_brock = OptimizationProblem(lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
	
	def test_rosen_brock(self):
		print("\nTesting Rosenbrock")
		solver = NewtonInexactLineSearch(self.rosen_brock, n=2)
		expected_solution = np.array([1.0, 1.0])
		
		# Assert check -> is result close enough to expected
		np.testing.assert_allclose(solver.solve(), expected_solution, atol=1e-1)
		path = np.array(solver.path)
		
		# Create a grid for the contour plot
		X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
		Z = self.rosen_brock.evaluate(np.array([X, Y]))
	
		# Create the contour plot
		contour = plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='jet')
		plt.plot(path[:, 0], path[:, 1], 'k-', linewidth=2)  # plot the optimization path
		plt.plot(path[:, 0], path[:, 1], 'ro')  # mark the points along the path with red dots
	
		# Save the plot to a file instead of displaying it
		plt.savefig('contour_plot.png')
		
# Run all unit tests
if __name__ == "__main__":
	unittest.main()
