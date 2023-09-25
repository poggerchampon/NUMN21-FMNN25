
from numpy import *
from matplotlib.pyplot import *
from .newton_inexact_line_search import NewtonInexactLineSearch 

class BFGS(NewtonInexactLineSearch):
    
    def __init__(self, opt_problem, n, h=1e-5, tolerance=1e-5, max_iterations=1000, initial_guess = None):
    
        super().__init__(self, opt_problem, n, h, tolerance, max_iterations, initial_guess)
                    # Start with zeros if no initial guess is specified
        x = self.initial_guess if self.initial_guess is not None else zeros(self.n)
        
        self.H = inv_approximate_hessian(self.opt_problem.objective_func, \
                                       self.opt_problem.gradient_func, x)
    
    def compute_direction(self, x, evaluate_func, gradient_func, current_gradient):
        
        s = -self.H @ current_gradient
        
        return s
        
    def update_hessian_invHessian(self):
        
        x_new = self.path[-1]
        x_old = self.path[-2]
        
        grad_new = self.opt_problem.gradient(x_new)
        grad_old = self.opt_problem.gradient(x_old)
        
        H = self.H    
        
        delta = x_new - x_old 
        gamma = grad_new - grad_old
        
        # Updating H
        
        self.H = H + (1 + (gamma.T @ H @ gamma)/(delta.T @ gamma))*((delta @ delta.T)*(1/(delta.T @ gamma))) \
            - (delta @ gamma.T @ H + H @ gamma @ delta.T)*(1/(delta.T @ gamma))
            
