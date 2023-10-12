
from numpy import *
from matplotlib.pyplot import *
from scipy.sparse.linalg import spsolve

# This is a way of constructing A using sparse. Use spsolve instead of solve. 
# Otherwise laplace_solver should work as before i think.  
 

# Constructing sparse version of A. Each number in the variables represents 
# the offset for the diagonal (m stands for minus).

diag_0 = array([-4]*(n*m))
diag_m1 = array([1]*(n*m))
diag_1 = array([1]*(n*m))
diag_mn = array([1]*(n*m)) 
diag_n = array([1]*(n*m))            

data = array([diag_mn, diag_m1, diag_0, diag_1, diag_n])
A_sparse = sp.spdiags(data,[-n, -1, 0, 1 , n],n*m, n*m,format = 'csr')


# Constructing b
b = zeros(n*m)

# Left boundary
b[::n] = u[:,0]

# Bottom boundary
b[n*m-n:] = u[-1,:]

# Top boundary
b[:n] = u[0,:]

# Right boundary
b[n-1::n] = u[:,-1]

return A_sparse,b
