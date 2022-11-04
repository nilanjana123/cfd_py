from sympy import Matrix, sin, cos
import sympy
import scipy
sympy.var( 'x, t' )
A = Matrix([[(sin(2-0.1*x)*sin(t)*x+cos(2-0.1*x)*cos(t)*x)*cos(3-0.1*x)*cos(t)],
            [(cos(2-0.1*x)*sin(t)*x+sin(2-0.1*x)*cos(t)*x)*sin(3-0.1*x)*cos(t)],
            [(cos(2-0.1*x)*sin(t)*x+cos(2-0.1*x)*sin(t)*x)*sin(3-0.1*x)*sin(t)]])

# integration intervals
x1,x2,t1,t2 = (30, 75, 0, 2*scipy.pi)

# element-wise integration
from sympy.utilities import lambdify
# from sympy.mpmath import quad
from scipy.integrate import dblquad
A_int1 = scipy.zeros( A.shape, dtype=float )
A_int2 = scipy.zeros( A.shape, dtype=float )
for (i,j), expr in scipy.ndenumerate(A):
    tmp = lambdify( (x,t), expr, 'math' )
    # A_int1[i,j] = quad( tmp, (x1, x2), (t1, t2) )
    # or (in scipy)
    A_int2[i,j] = dblquad( tmp, t1, t2, lambda x:x1, lambda x:x2 )[0]
