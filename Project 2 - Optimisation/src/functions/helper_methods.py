import numpy as np

# Default numerical method if gradient is not specified
def numerical_gradient(evaluate_func, x, h=1e-5):
    grad = np.zeros(len(x), dtype=float)
    identity_matrix=np.eye(len(x))
    
    for i in range(len(x)):
        grad[i] = (evaluate_func(x + h * identity_matrix[i]) - evaluate_func(x - h * identity_matrix[i])) / (2.0 * h)
    
    return grad

# Returns the inverse of hessian
def inv_approximate_hessian(evaluate_func, gradient_func, x, n, h=1e-5):
    return np.linalg.inv(approximate_hessian(evaluate_func, gradient_func, x, n, h))

def approximate_hessian(evaluate_func, gradient_func, x, n, h=1e-5):
    hessian = np.zeros((n, n))
    identity_matrix = np.eye(n)
    
    for i in range(n):
        # Diagonal elements
        x_i_pos = x + h * identity_matrix[i]
        x_i_neg = x - h * identity_matrix[i]
        f_i_pos = evaluate_func(x_i_pos)
        f_i_neg = evaluate_func(x_i_neg)
        hessian[i][i] = (f_i_pos - 2 * evaluate_func(x) + f_i_neg) / (h**2)
        
        # Start from i + 1 to avoid recomputing diagonal, and use symmetry
        for j in range(i+1, n):
            x_ij_pos = x + h * (identity_matrix[i] + identity_matrix[j])
            x_ij_neg = x - h * (identity_matrix[i] + identity_matrix[j])
            x_i_pos_j_neg = x + h * identity_matrix[i] - h * identity_matrix[j]
            x_i_neg_j_pos = x - h * identity_matrix[i] + h * identity_matrix[j]
            
            f_ij_pos = evaluate_func(x_ij_pos)
            f_ij_neg = evaluate_func(x_ij_neg)
            f_i_pos_j_neg = evaluate_func(x_i_pos_j_neg)
            f_i_neg_j_pos = evaluate_func(x_i_neg_j_pos)
            
            hessian[i][j] = (f_ij_pos - f_i_pos_j_neg - f_i_neg_j_pos + f_ij_neg) / (4 * h**2)
            hessian[j][i] = hessian[i][j] 
            
    return hessian


