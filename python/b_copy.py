import numpy as np
from sympy import *
var('x l A B M')

rc = 10
t = 10

# n1 = 1 - (3*x)/l + (2*x**2)/l**2
# n2 = 4*x - (4*x**2)/l**2
# n3 = (-x)/l + (2*x**2)/l**2
n1 = 1 - (3*x) + (2*x**2)
n2 = 4*x - (4*x**2)
n3 = (-x) + (2*x**2)

A = Matrix([n1 ,n2, n3])
print(A)

dA = A.diff(x)
print(dA)

B = Transpose(A).T
print(B.T)

dB = Transpose(dA).T
print(dB.T)

M = A.multiply(B.T)
print(M)

K = dA.multiply(dB.T)
print(K)

MI = M.integrate((x,0,1))
print(MI)

KI = K.integrate((x,0,1))
print(KI)
kin = np.array(KI).astype(np.float64)
kin = kin*3
print(kin.reshape(1,9))
min = np.array(MI).astype(np.float64)
min = min*30
print(min)


# import numpy as np
# from sympy import *

na = 3
# arr = [1,2,3,4,5,6,7,8,9]
# arrM = [4,2,-1,2,16,2,-1,2,4]
# arrK = [-7,8,-1,8,16,8,-1,8,-7]
# # m = np.array(arrM, dtype = np.float64).reshape(na,na)
# # k = np.array(arrK, dtype = np.float64).reshape(na,na)
# # a = np.array(arr, dtype = np.float64).reshape(na,na)

m = np.array(min , dtype=np.float64)
k = np.array(kin , dtype=np.float64)
c = np.array([100,100,100], dtype = np.float64)
# m = np.array([1,2,3,3,5,6,7,1,9],dtype=np.float64).reshape(na,na)
# k = np.array([1,2,3,3,5,6,7,1,9],dtype=np.float64).reshape(na,na)

# print(a)
# print(m)
# print(k)

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
# matK = -1*matK
C = c_large(c,rc)
# print(matA)
# print(C)
print(matM)
print(matK)

dx = 1/(rc-1)
dt = 0.05
D = 0.03

#lhs
'''
def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/30)*matM + ((D*dt)/(3*dx))*matK
    mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs = (dx/30) * np.matmul(matM,C)

    new_C = np.matmul(mat_lhs_inv,mat_rhs)
    new_C[0] = 0 #
    new_C[rc-1] = 0 #
   
    # print(mat_lhs_inv)
    # print(mat_rhs)
    return(new_C)
'''

def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    # mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)

    # new_C = np.copy(C)
    x = np.copy(C)
    x[0] = 0
    x[rc-1] = 0
    for r in range(20):
    # for onr in range(rc):
        for i in range(1,rc-1):
            mat1 = mat_lhs[i]
            mat1 = np.delete(mat1,i)
            px = np.delete(x,i)
            res = 0
            for k in range(rc-1):
                res += mat1[k]*px[k]
            x[i] = (mat_rhs[i] - res)/mat_lhs[i][i]
            # print(x)
    #   print(x.round(decimals=2))
    ret_c = np.copy(x)
    ret_c[0] = 0
    ret_c[rc-1] = 0
    # print(ret_c.round(2))
    return(ret_c)
        
# list_new_C = np.ndarray()
def new_C_repeat():
    new_C = C
    for i in range(t):
        new_C = new_C_calc(matM,matK,dx,dt,D,new_C)
        print(new_C.round(2))
        # list_new_C.append(new_C)

list_new_C = new_C_repeat()
# print(list_new_C)