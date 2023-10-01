import unittest
import numpy.testing as npt

from src.functions import helper_methods
from src.optimization_problem import OptimizationProblem

# Unit tests for OptimizationProblem class
class TestOptimizationProblem(unittest.TestCase):
	
	# Test evaluating 3^2 = 9
	def test_evaluate(self):
		def quadratic_function(x):
			return x**2
		
		opt_problem = OptimizationProblem(quadratic_function)
		self.assertEqual(opt_problem.evaluate(3), 9)
	
	# Test evaluation gradient of x^2 (initial specified gradient)
	def test_gradient_provided(self):
		def quadratic_gradient(x):
			return 2*x
		
		opt_problem = OptimizationProblem(lambda x: x**2, quadratic_gradient)
		self.assertEqual(opt_problem.gradient(3), 6)
		
	# Test evaluation gradient of x^2 (tensorflow)
	def test_gradient_tensorflow(self):
		opt_problem = OptimizationProblem(lambda x: x**2)
		self.assertEqual(opt_problem.gradient(3, 'tf'), 6)
	
	# Test numerical gradient
	def test_gradient_not_provided(self):
		opt_problem = OptimizationProblem(lambda x: x**2)
		npt.assert_almost_equal(opt_problem.gradient([3], 'numerical'), [6])
		
	# Test getting num of parameters for multivariable
	def test_get_num_of_params(self):
		opt_problem = OptimizationProblem(lambda x, y, z: x * y + z)
		self.assertEqual(opt_problem.get_num_of_parameters(), 3)
			
# Run all unit tests
if __name__ == "__main__":
	unittest.main()
