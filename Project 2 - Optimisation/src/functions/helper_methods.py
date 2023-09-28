import numpy as np

# Default numerical method if gradient is not specified
def numerical_gradient(evaluate_func, x, h=1e-5):
    grad = np.zeros(len(x), dtype=float)
    identity_matrix=np.eye(len(x))
    
    for i in range(len(x)):
        grad[i] = (evaluate_func(x + h * identity_matrix[i]) - evaluate_func(x - h * identity_matrix[i])) / (2.0 * h)
    
    return np.array(grad)

# Returns the inverse of hessian
def inv_approximate_hessian(gradient_func, x, n, h=1e-5):
    return np.linalg.inv(approximate_hessian(gradient_func, x, n, h))

# numerical_hessian() misses off-diagonal elements, also misses last row because of
# 'dim-1' in the loop
def approximate_hessian(gradient_func, x, n, h=1e-5):
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
