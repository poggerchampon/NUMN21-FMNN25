import unittest
import numpy as np

from src.functions import approximate_hessian
from src.optimization_problem import OptimizationProblem

class TestApproximateHessian(unittest.TestCase):
	
	# f(x) = x^2 + y^2 --> hessian should be [[2, 0], [0, 2]]
	def test_basic_functionality(self):
		opt_problem = OptimizationProblem(lambda x: x[0]**2 + x[1]**2)
		
		hessian = approximate_hessian(opt_problem.get_gradient(), [1, 1])
		expected = np.array([[2, 0], [0, 2]])
		np.testing.assert_array_almost_equal(hessian, expected, decimal=2)
		
	def test_output_size(self):
		opt_problem = OptimizationProblem(lambda x: x[0]**2 + x[1]**2)
		
		hessian = approximate_hessian(opt_problem.get_gradient(), [1, 1])
		self.assertEqual(hessian.shape, (2, 2))
		
# Run all unit tests
if __name__ == "__main__":
	unittest.main()
