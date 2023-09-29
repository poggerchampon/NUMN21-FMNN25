import numpy as np

from .newton_exact_line_search import NewtonExactLineSearch 
from src.functions import inv_approximate_hessian

class BFGS(NewtonExactLineSearch):
    
    def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-5, max_iterations=1000, initial_guess = None):
        super().__init__(opt_problem, n, h, tolerance, max_iterations, initial_guess)
        # Start with zeros if no initial guess is specified
        x = self.initial_guess if self.initial_guess is not None else np.zeros(self.n)
        self.H_inv = inv_approximate_hessian(self.opt_problem.gradient_func, x, n)
    
    def compute_direction(self, x, gradient_func, current_gradient):
        return -np.dot(self.H_inv, current_gradient)
        
    def update_inv_hessian(self, x, gradient_func, current_gradient):
        if self.prev_x is not None and self.prev_gradient is not None:
            delta_x = (x - self.prev_x).reshape(-1, 1) 
            delta_g = (current_gradient - self.prev_gradient).reshape(-1, 1) 
            
            # ik this is unreadable, but so is the math
            H_inv_g = np.dot(self.H_inv, delta_g)
            g_H_inv = np.dot(delta_g.T, self.H_inv)
            
            first_term_1 = 1 + (np.dot(delta_g, H_inv_g) / np.dot(delta_x.T, delta_g))
            first_term_2 = np.dot(delta_x, delta_x.T) / np.dot(delta_x.T, delta_g)
            first_term = np.dot(first_term_1, first_term_2)
            
            second_term_numerator = np.dot(delta_x, g_H_inv) + np.dot(H_inv_g, delta_x.T)
            second_term_denominator = np.dot(delta_x.T, delta_g)
            second_term = second_term_numerator / second_term_denominator
            
            
            self.H_inv += (first_term - second_term)
