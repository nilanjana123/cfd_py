import numpy as np
from sympy import *

na = 3
arr = [1,2,3,4,5,6,7,8,9]
arrM = [4,2,-1,2,16,2,-1,2,4]
arrK = [-7,8,-1,8,16,8,-1,8,-7]
m = np.array(arrM, dtype = np.float64).reshape(na,na)
k = np.array(arrK, dtype = np.float64).reshape(na,na)
a = np.array(arr, dtype = np.float64).reshape(na,na)

c = np.array([100,100,100], dtype = np.float64)

# print(a)
# print(m)
# print(k)
rc = 9


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

matA = mat_large(a,rc)
matM = mat_large(m,rc)
matK = mat_large(k,rc)

C = c_large(c,rc)
# print(matA)
# print(C)
print(matM)
print(matK)

dx = 0.1
dt = 0.5
D = 1.64*10**(-4)

#lhs
def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)

    # new_C = np.copy(C)
    x = np.copy(C)
    for r in range(10):
    # for onr in range(rc):
        for i in range(rc):
            mat1 = mat_lhs[i]
            mat1 = np.delete(mat1,i)
            px = np.delete(x,i)
            # print(px.round(2))
            # print(x.round(2))
            # d = x[i] if x[i]>0 else 1.0
            # px.transpose()
            # print(px)
            # print(mat[i])
            # print(mat1)
            # x[i] = (np.matmul(mat1,px))/mat_lhs[i][i]# + bias[i])/d
            res = 0
            for k in range(rc-1):
                res += mat1[k]*px[k]
            x[i] = (mat_rhs[i] - res)/mat_lhs[i][i]
            # print(x)
    #   print(x.round(decimals=2))
    ret_c = np.copy(x)
    # print(ret_c.round(2))
    return(ret_c)
    
# list_new_C = np.ndarray()
def new_C_repeat():
    new_C = np.copy(C)
    for i in range(10):
        new_C = new_C_calc(matM,matK,dx,dt,D,new_C)
        print(new_C.round(2))
        # list_new_C.append(new_C)

list_new_C = new_C_repeat()
# print(list_new_C)