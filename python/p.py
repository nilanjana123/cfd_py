import numpy as np
from sympy import *

na = 3
# arr = [1,2,3,4,5,6,7,8,9]
arrM = [4,2,-1,2,16,2,-1,2,4]
arrK = [-7,8,-1,8,16,8,-1,8,-7]

# arrM = [2,1,1,2]
# arrK = [1,-1,-1,1]
m = np.array(arrM, dtype = np.float64).reshape(na,na)
k = np.array(arrK, dtype = np.float64).reshape(na,na)
# a = np.array(arr, dtype = np.float64).reshape(na,na)

c = np.array([100,100,100], dtype = np.float64)

# print(a)
# print(m)
# print(k)
rc = 101


def mat_large(a,rc):
    mat = np.array([0]*(rc*rc),dtype = np.float64).reshape(rc,rc)
    for k in range(0,rc-2,1):
        for i in range(na):
            for j in range(na):
                mat[i+k][j+k] += a[i][j]
    return mat

def c_large(c,rc):
    C = np.array([0]*(rc),dtype = np.float64)
    for k in range(0,rc-2,1):
        for i in range(na):
            C[i+k] = c[i]   
        # C[0] = 0 #
        # C[rc-1] = 0 #
    return C

# matA = mat_large(a,rc)
matM = mat_large(m,rc)
matK = mat_large(k,rc)

C = c_large(c,rc)
# print(matA)
# print(C)
# print(matM)
# print(matK)

dx = 0.01
dt = 0.001
D = 0.03
# dx = 1/9
# dt = 0.05
# D = 1.64*10**(-4)

#lhs
def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)

    new_C = np.matmul(mat_lhs_inv,mat_rhs)
    
    # mat_lhs =  (dx/6)*matM + ((D)/(dx))*matK
    # mat_lhs_inv = np.linalg.inv(mat_lhs)

    # mat_rhs_part = ((dx/(6*dt))*matM - (D/dx)*matK) 

    # mat_rhs = np.matmul(mat_rhs_part, C)

    # new_C = np.matmul(mat_lhs_inv,mat_rhs)

    new_C[0] = 0 #
    new_C[rc-1] = 0 #
   
    # print(mat_lhs_inv)
    # print(mat_rhs)
    return(new_C)
    
# list_new_C = np.ndarray()
def new_C_repeat():
    new_C = np.copy(C)
    for i in range(10):
        new_C = new_C_calc(matM,matK,dx,dt,D,new_C)
        print(new_C.round(2))
        # list_new_C.append(new_C)

list_new_C = new_C_repeat()
# print(list_new_C)