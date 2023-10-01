import unittest
import numpy as np

from src.optimization_methods import GoodBroyden, BadBroyden, SymmetricBroyden, DFP, BFGS
from src.optimization_problem import OptimizationProblem
from src.functions import save_rosenbrock_plot

class TestQuasiNewton(unittest.TestCase):
	def setUp(self):
		self.rosen_brock = OptimizationProblem(lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
		self.expected_solution = expected_solution = np.array([1.0, 1.0])

	def test_good_broyden(self):
		print("\nTesting Good Broyden")
		solver = GoodBroyden(self.rosen_brock, n=2, initial_guess=np.array([0.0, 0.0]), progress_bar=True)
		result = solver.solve()
			
		# Assert check -> is result close enough to expected
		np.testing.assert_allclose(result, self.expected_solution, atol=1e-1)
		
		# Save plot of the optimisation path
		save_rosenbrock_plot(np.array(solver.path), filename='good_broyden_contour')
		
	def test_bad_broyden(self):
		print("\nTesting Bad Broyden")
		solver = BadBroyden(self.rosen_brock, n=2, tolerance=1e-2, initial_guess=np.array([0.0, 0.0]), progress_bar=True)
		result = solver.solve()
		
		# Assert check -> is result close enough to expected
		np.testing.assert_allclose(result, self.expected_solution, atol=2)
		
	def test_symmetric_broyden(self):
		print("\nTesting Symmetric Broyden")
		solver = SymmetricBroyden(self.rosen_brock, n=2, tolerance=1e-2, initial_guess=np.array([0.0, 0.0]), progress_bar=True)
		result = solver.solve()
		
		# Assert check -> is result close enough to expected
		np.testing.assert_allclose(result, self.expected_solution, atol=0.2)
		
		# Save plot of the optimisation path
		save_rosenbrock_plot(np.array(solver.path), filename='symmetric_broyden_contour')
		
	def test_dfp(self):
		print("\nTesting DFP")
		solver = DFP(self.rosen_brock, n=2, tolerance=1e-2, initial_guess=np.array([0.0, 0.0]), progress_bar=True)
		result = solver.solve()
		
		# Assert check -> is result close enough to expected
		np.testing.assert_allclose(result, self.expected_solution, atol=1e-1)
		
		# Save plot of the optimisation path
		save_rosenbrock_plot(np.array(solver.path), filename='dfp_contour')
		
	def test_bfgs(self):
		print("\nTesting BFGS")
		solver = BFGS(self.rosen_brock, n=2, initial_guess=np.array([0.0, 0.0]), progress_bar=True)
		result = solver.solve()
		
		# Assert check -> is result close enough to expected
		np.testing.assert_allclose(result, self.expected_solution, atol=1e-1)
		
		# Save plot of the optimisation path
		save_rosenbrock_plot(np.array(solver.path), filename='bfgs_contour')
		
# Run all unit tests
if __name__ == "__main__":
	unittest.main()
	
