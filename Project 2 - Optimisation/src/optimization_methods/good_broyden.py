import numpy as np

from .newton_inexact_line_search import NewtonInexactLineSearch
from src.functions import inv_approximate_hessian

class GoodBroyden(NewtonInexactLineSearch):
    
    def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-3, max_iterations=1000, initial_guess=None, progress_bar=False):
        super().__init__(opt_problem, n, h, tolerance, max_iterations, initial_guess, progress_bar)

        self.H_inv = np.eye(n) # Initialise inverse hessian (identity matrix)
        
    def _compute_direction(self, x, gradient_func, current_gradient):
        return -np.dot(self.H_inv, current_gradient)
        
    def _update_inv_hessian(self, x, gradient_func, current_gradient):
        if self.prev_x is not None and self.prev_gradient is not None:
            delta_x = x - self.prev_x
            delta_g = current_gradient - self.prev_gradient

            deltax_Hinv = np.dot(delta_x,H_inv) 
            numerator = np.dot(delta_x - np.dot(H_inv,delta_g),deltax_Hinv)
            denominator = np.dot(deltax_Hinv,delta_g)

            if denominator != 0:
                self.H_inv += numerator/denominator
            else:
                print("Division by zero encountered in update_inv_hessian, skipping this update")
                
    def solve(self):
        self.prev_x = None
        self.prev_gradient = None
        return super().solve()
