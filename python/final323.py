import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import sys
import time 

start = time.time()
var('x l A B M')
na = 3
rc = 101
st  = 20
t = 200
dx = 1/(rc-1)
dt = 0.05
D  = 0.00003

epsilon = 1/(10**(10))

def form_MK_small():

    n1 = 1 - (3*x) + (2*x**2)
    n2 = 4*x - (4*x**2)
    n3 = (-x) + (2*x**2)

    A = Matrix([n1 ,n2, n3])
    dA = A.diff(x)
    B = Transpose(A).T
    dB = Transpose(dA).T
    M = A.multiply(B.T)
    K = dA.multiply(dB.T)
    MI = M.integrate((x,0,1))
    KI = K.integrate((x,0,1))
    kin = np.array(KI).astype(np.float64)
    kin = kin*3
    min = np.array(MI).astype(np.float64)
    min = min*30
    m = np.array(min , dtype=np.float64)
    k = np.array(kin , dtype=np.float64)
    c = np.array([100,100,100], dtype = np.float64)

    return m, k , c

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
    return C

def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    # mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)

    # new_C = np.copy(C)
    x = np.copy(C)
    x[0] = 0
    x[rc-1] = 0
    for r in range(st):
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


def new_C_repeat(arr):
    new_C = new_C_calc(matM,matK,dx,dt,D,C)
    for i in range(t):
        new_C = new_C_calc(matM,matK,dx,dt,D,new_C)
        # arr = np.append(arr, np.array(new_C),axis=0)
        arr.append(new_C)
        # print(new_C.round(2))
        if i%10 == 0:
            xax = np.arange(rc)
            xax = xax/100
            plt.plot(xax,new_C)
            # plt.pause(0.01)

m,k,c = form_MK_small()
matM = mat_large(m,rc)
matK = mat_large(k,rc)
C = c_large(c,rc)
arr = []

list_new_C = new_C_repeat(arr)
print(arr)

plt.xlim([0,1.1])
plt.ylim([0, 110])
str1 = "Time evolution of concentration at D = " + format(D)
plt.title(str1)
plt.xlabel("Length x")
plt.ylabel("Concentration space C(t,x)")
end = time.time()
print("Execution time of the program is- ", end-start)
# plt.show()
fname = format(D)+"P.png"
plt.savefig(fname)
