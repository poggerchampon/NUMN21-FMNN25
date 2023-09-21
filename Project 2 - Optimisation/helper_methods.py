import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import importlib.util

# Default numerical method if gradient is not specified
def numerical_gradient(evaluate_func, x, h=1e-5):
    return (evaluate_func(x + h) - evaluate_func(x - h)) / (2.0 * h)

# added n since no access to self object anymore
def numerical_hessian(evaluate_func, x, n, h=1e-5):
    dim = n
    hessian=np.zeros((dim,dim))
    identity_matrix=np.diag(np.ones(dim))

    for i in range(0,dim-1):
        for j in range(0,dim-1):
        
            hessian[i][j]=(evaluate_func(x+h*identity_matrix[i])+evaluate_func(x+h*identity_matrix[j])-2*evaluate_func(x))/h**2

    return hessian

def inv_numerical_hessian(evaluate_func, x, n, h=1e-5):
    return np.linalg.inv(numerical_hessian(evaluate_func, x, n, h))

# Plot function with specified x_range and number of points
def plot_objective_2D(evaluate_func, range_x = None, num_points=100):
    if range_x is None:
        print("Range for x is not specified")
        return
    
    x_values = np.linspace(range_x[0], range_x[1], num_points)
    y_values = [evaluate_func(x) for x in x_values]
    
    plt.plot(x_values, y_values)
    plt.xlabel("x")
    plt.ylabel("Objective function value")
    plt.title("Objective function")
    plt.show()

# 3D visualisation?
def plot_objective_3D(self):
    return
