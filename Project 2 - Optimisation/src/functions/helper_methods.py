import numpy as np

# Default numerical method if gradient is not specified
def numerical_gradient(evaluate_func, x, h=1e-5):
    return (evaluate_func(x + h) - evaluate_func(x - h)) / (2.0 * h)

def numerical_hessian(evaluate_func, x, n, h=1e-5):
    dim = len(x)
    hessian=np.zeros((dim,dim))
    identity_matrix=np.diag(np.ones(dim))

    for i in range(0,dim-1):
        for j in range(0,dim-1):
        
            hessian[i][j]=(evaluate_func(x+h*identity_matrix[i])+evaluate_func(x+h*identity_matrix[j])-2*evaluate_func(x))/h**2

    return hessian

# Returns the inverse of hessian
def inv_approximate_hessian(gradient_func, x, h=1e-5):
    return np.linalg.inv(approximate_hessian(gradient_func, x, h))

# numerical_hessian() misses off-diagonal elements, also misses last row because of
# 'dim-1' in the loop
def approximate_hessian(gradient_func, x, h=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))
    identity_matrix = np.eye(n)
    base_gradient = gradient_func(x)
    
    for i in range(n):
        # Diagonal elements
        x_i_pos = x + h * identity_matrix[i]
        gradient_i_pos = gradient_func(x_i_pos)
        hessian[i][i] = (gradient_i_pos[i] - base_gradient[i]) / h
        
        # Start from i + 1 to avoid recomputing diagonal, and use symmetry
        for j in range(i+1, n):
            x_j_pos = x + h * identity_matrix[j]
            gradient_j_pos = gradient_func(x_j_pos)
            
            # Take advantage of symmetry to compute both sides of the matrix
            value = (gradient_j_pos[i] - base_gradient[i]) / h
            hessian[i][j] = value
            hessian[j][i] = value
            
    return hessian
