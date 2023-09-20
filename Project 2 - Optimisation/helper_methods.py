
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import importlib.util



# Default numerical method if gradient is not specified
def numerical_gradient(self, x, h=1e-5):
    return (self.evaluate(x + h) - self.evaluate(x - h)) / (2.0 * h)

def numerical_hessian(self, x, h=1e-5):
    
    dim=len(self.inspect.signature(self.objective_func))
    hessian=np.zeros((dim,dim))
    identity_matrix=np.diag(np.ones(dim))

    for iy in range(0,dim-1):
        for j in range(0,dim-1):
        
            hessian[i][j]=(self.evaluate(x+h*identity_matrix[i])+self.evaluate(x+h*identity_matrix[j])-2*self.evaluate(x))/h**2

    return hessian


def inv_numerical_hessian(self,x,h=1e-5):

    return np.linalg.inv(numerical_hessian(self,x,h))



# Plot function with specified x_range and number of points
def plot_objective_2D(self, range_x = None, num_points=100):
    if range_x is None:
        print("Range for x is not specified")
        return
    
    x_values = np.linspace(range_x[0], range_x[1], num_points)
    y_values = [self.evaluate(x) for x in x_values]
    
    plt.plot(x_values, y_values)
    plt.xlabel("x")
    plt.ylabel("Objective function value")
    plt.title("Objective function")
    plt.show()

# 3D visualisation?
def plot_objective_3D(self):
    return
    

