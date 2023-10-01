import numpy as np

from .newton_inexact_line_search import NewtonInexactLineSearch 
from src.functions import inv_approximate_hessian

class DFP(NewtonInexactLineSearch):
    
    def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-5, max_iterations=2000, initial_guess = None):
        super().__init__(opt_problem, n, h, tolerance, max_iterations, initial_guess)
        # Start with zeros if no initial guess is specified
        x = self.initial_guess if self.initial_guess is not None else np.zeros(self.n)
        self.H_inv = inv_approximate_hessian(self.opt_problem.get_gradient(), x, n)
    
    def compute_direction(self, x, gradient_func, current_gradient):
        return -np.dot(self.H_inv, current_gradient)
        
    def update_inv_hessian(self, x, gradient_func, current_gradient):
        if self.prev_x is not None and self.prev_gradient is not None:
            delta_x = (x - self.prev_x).reshape(-1, 1) 
            delta_g = (current_gradient - self.prev_gradient).reshape(-1, 1) 
            
            first_term = np.dot(delta_x, delta_x.T) / np.dot(delta_x.T, delta_g)
            
            H_inv_g = np.dot(self.H_inv, delta_g)
            g_H_inv = np.dot(delta_g.T, self.H_inv)
            second_term = np.dot(H_inv_g, g_H_inv) / np.dot(delta_g.T, H_inv_g)
            
            self.H_inv += (first_term - second_term)

    def solve(self):
        self.prev_x = None
        self.prev_gradient = None
        return super().solve()
