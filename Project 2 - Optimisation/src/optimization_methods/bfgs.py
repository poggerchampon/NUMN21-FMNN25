import numpy as np

from .newton_inexact_line_search import NewtonInexactLineSearch 
from src.functions import inv_approximate_hessian

class BFGS(NewtonInexactLineSearch):
    
    def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-3, max_iterations=1000, initial_guess=None, progress_bar=False):
        super().__init__(opt_problem, n, h, tolerance, max_iterations, initial_guess, progress_bar)
        self.H_inv = np.eye(n) # Initialise inverse hessian (identity matrix)
        
    def _compute_direction(self, x, gradient_func, current_gradient):
        return -np.dot(self.H_inv, current_gradient)
        
    def _update_inv_hessian(self, x, gradient_func, current_gradient):
        if self.prev_x is not None and self.prev_gradient is not None:
            delta_x = (x - self.prev_x).reshape(-1, 1) 
            delta_g = (current_gradient - self.prev_gradient).reshape(-1, 1) 
        
            # Update H according to math
            H_inv_g = np.dot(self.H_inv, delta_g)
            g_H_inv = np.dot(delta_g.T, self.H_inv)
            
            first_term_1 = 1 + (np.dot(delta_g.T, H_inv_g) / np.dot(delta_x.T, delta_g))
            first_term_2 = np.dot(delta_x, delta_x.T) / np.dot(delta_x.T, delta_g)
            first_term = first_term_1 * first_term_2
            
            second_term_numerator = np.dot(delta_x, g_H_inv) + np.dot(H_inv_g, delta_x.T)
            second_term_denominator = np.dot(delta_x.T, delta_g)
            second_term = second_term_numerator / second_term_denominator
            
            self.H_inv += (first_term - second_term)
            
    def get_current_hessian_approximation(self):
        return self.H_inv
            

    def solve(self):
        self.prev_x = None
        self.prev_gradient = None
        return super().solve()
            
