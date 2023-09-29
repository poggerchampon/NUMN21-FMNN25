import numpy as np

from .newton_inexact_line_search import NewtonInexactLineSearch
from src.functions import inv_approximate_hessian

class BadBroyden(NewtonInexactLineSearch):
    def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-2, max_iterations=1000, initial_guess = None):
        super().__init__(opt_problem, n, h, tolerance, max_iterations, initial_guess)
        
        # Start with zeros if no initial guess is specified
        x = self.initial_guess if self.initial_guess is not None else np.zeros(self.n)
        self.H_inv = inv_approximate_hessian(self.opt_problem.get_gradient(), x, n)
    
    def compute_direction(self, x, gradient_func, current_gradient):
        return -np.dot(self.H_inv, current_gradient)
        
    def update_inv_hessian(self, x, gradient_func, current_gradient):
        if self.prev_x is not None and self.prev_gradient is not None:
            delta_x = x - self.prev_x 
            delta_g = current_gradient - self.prev_gradient
        
            H_inv_gamma = np.dot(self.H_inv, delta_g)
            numerator = np.dot((delta_x - H_inv_gamma), delta_g.T)
            denominator = np.dot(delta_g.T, delta_g)
        
            if denominator != 0:
                self.H_inv += numerator / denominator
            else:
                print("Division by zero encountered in update_inv_hessian, skipping this update")
        
    def solve(self):
        self.prev_x = None
        self.prev_gradient = None
        return super().solve()
