import unittest
import numpy as np

from src.solver.laplace_solver import solve_laplace

class TestLaplaceSolver(unittest.TestCase):
  
    def test_boundary_conditions(self):
      # Create a 5x5 grid with boundary conditions
      u = np.array([
        [15, 15, 15, 15, 15],
        [15, 0, 0, 0, 40],
        [15, 0, 0, 0, 5],
        [15, 0, 0, 0, 5],
        [15, 15, 15, 15, 15]
      ])
      
      u_new = solve_laplace(u)
      print(f"\nChecking preserved boundaries. Following is the updated room\n {u_new}")
      
      # Check that the boundary conditions are preserved
      self.assertTrue(np.all(u_new[0, :] == 15))
      self.assertTrue(np.all(u_new[-1, :] == 15))
      self.assertTrue(np.all(u_new[:, 0] == 15))
      self.assertTrue(np.all(u_new[:, -1] == u[:, -1]))
      
    def test_output_shape(self):
      # Create a 4x4 grid with arbitrary boundary conditions
      u = np.array([
        [10, 10, 10, 10],
        [10, 0, 0, 20],
        [10, 0, 0, 5],
        [10, 10, 10, 10]
      ])
      
      u_new = solve_laplace(u)
      print(f"\nChecking output shape. Following is the updated room\n {u_new}")
      
      # Check that the output shape matches the input shape
      self.assertEqual(u_new.shape, u.shape)

if __name__ == '__main__':
    unittest.main()
  
