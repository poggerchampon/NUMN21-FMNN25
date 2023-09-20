import unittest
from optimization_problem import OptimizationProblem

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
	
	# Test exception raise for evaluating non specified gradient
	def test_gradient_not_provided(self):
		opt_problem = OptimizationProblem(lambda x: x**2)
		with self.assertRaises(NotImplementedError):
			opt_problem.gradient(3)
	


# Run all unit tests
if __name__ == "__main__":
	unittest.main()