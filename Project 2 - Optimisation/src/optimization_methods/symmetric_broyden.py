import numpy as np

from .newton_exact_line_search import NewtonExactLineSearch 
from src.functions import inv_approximate_hessian

class SymmetricBroyden(NewtonExactLineSearch):
    
    def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-2, max_iterations=1500, initial_guess = None):
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
            
            u = delta_x - np.dot(self.H_inv, delta_g)
            a = 1/(np.dot(u.T, delta_g))
            
            # Updating H
            self.H_inv += a * np.dot(u, u.T)
