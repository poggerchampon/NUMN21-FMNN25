import unittest
import numpy as np

from src.functions import approximate_hessian, numerical_gradient
from src.optimization_problem import OptimizationProblem

class TestApproximateHessian(unittest.TestCase):
	
	def setUp(self):
		f2 = lambda x: x[0]**2 + x[1]**2
		f3 = lambda x: x[0]**2 + x[1]**2 + x[2]**2
		self.quad_2D = OptimizationProblem(f2)
		self.quad_3D = OptimizationProblem(f3)
		
	def test_numerical_gradient(self):
		function = lambda x: np.array(x[0]**2, x[1]**2)
		
		grad = numerical_gradient(self.quad_2D.get_evaluate(), [1, 1])
		np.testing.assert_almost_equal(grad, [2, 2])
		
		grad = numerical_gradient(self.quad_3D.get_evaluate(), [1, 1, 1])
		np.testing.assert_almost_equal(grad, [2, 2, 2])
		
	# f(x) = x^2 + y^2 --> hessian should be [[2, 0], [0, 2]]
	def test_approximate_hessian(self):
		hessian = approximate_hessian(self.quad_2D.get_gradient(), np.array([1, 1]), 2)
		expected = np.array([[2, 0], [0, 2]])
		
		np.testing.assert_array_almost_equal(hessian, expected, decimal=2)
		
	def test_hessian_output_size(self):
		hessian = approximate_hessian(self.quad_2D.get_gradient(), np.array([1, 1]), 2)
		self.assertEqual(hessian.shape, (2, 2))
		
# Run all unit tests
if __name__ == "__main__":
	unittest.main()
