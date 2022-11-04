import numpy as np
from sympy import *
import matplotlib.pyplot as plt

var('x l A B M')
na = 3
rc = 11
st  = 100
t = 50
dx = 1/(rc-1)
dt = 0.05
D  = 0.03
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
'''
def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)
    x = np.copy(C)
    x[0] = 0
    x[rc-1] = 0
    for r in range(20):
        for i in range(1,rc-1):
            mat1 = mat_lhs[i]
            mat1 = np.delete(mat1,i)
            px = np.delete(x,i)
            res = 0
            for k in range(rc-1):
                res += mat1[k]*px[k]
            x[i] = (mat_rhs[i] - res)/mat_lhs[i][i]
    ret_c = np.copy(x)
    ret_c[0] = 0
    ret_c[rc-1] = 0
    return(ret_c)
'''
'''
def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)
    
    x = np.array([0]*rc ,dtype = np.float64)
    x_old = np.copy(x)
    for r in range(10):
        x[0] = 0
        for i in range(1,rc-1):
            mat1 = mat_lhs[i]
            mat1 = np.delete(mat1,i)
            px = np.delete(x,i)
            res = 0
            for k in range(rc-1):
                res += mat1[k]*px[k]
            x[i] = (mat_rhs[i] - res)/mat_lhs[i][i]
        x[rc-1] = 0
        dx = np.sqrt(np.dot(x-x_old, x-x_old))
        if dx < epsilon:
        # converged = True
            print('Converged!')
            break
    # assign the latest x value to the old value
        x_old = x
    ret_c = np.copy(x)
    return(ret_c)
'''

def seidel(a,x,b):
    x = np.array([0]*rc , dtype =np.float64)
    x_old = np.array([0]*rc , dtype =np.float64)
    for k in range(st):
        for i in range(0,rc):
            d = b[i]
            sum = 0
            for j in range(rc):
                if i!=j : 
                    sum += a[i][j]*x[j]
            x[i] = (b[i] - sum)/a[i][i]
        x[0] = 0
        x[rc-1] =0
        dx = np.dot(x-x_old,x-x_old)
        if  dx < epsilon:
            break
        x_old = x
    return x  

def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)
    x = np.array([0]*rc , dtype =np.float64)

    x = seidel(mat_lhs,x,mat_rhs)
    return(x)

'''
def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs = (dx/(30*dt))*matM + (D/(3*dx))*matK
    mat_rhs = (dx/(30*dt)) * np.matmul(matM,C)
    LD = np.array([0]*(rc*rc),dtype=np.float64).reshape(rc,rc)
    U = np.array([0]*(rc*rc),dtype=np.float64).reshape(rc,rc)
    # x = np.copy(C)
    x = np.array([0]*rc,dtype = np.float64)
    x[0] = 0
    x[rc-1] = 0
    for i in range(rc):
        for j in range(rc):
            if i>=j:
                LD[i][j] = mat_lhs[i][j]
            if i<j:
                U[i][j] = mat_lhs[i][j]
    LD_inv = np.linalg.inv(LD)
    LDB = np.matmul(LD_inv,mat_rhs)
    LDU = np.matmul(LD_inv,U)
    LDUX = np.matmul(LDU,x)
    LDUX = (-1)*LDUX
    x = np.add(LDUX,LDB)
    x[0] = 0
    x[rc-1] = 0
    return(x)
'''

def new_C_repeat():
    new_C = new_C_calc(matM,matK,dx,dt,D,C)
        # new_C = C
    for i in range(t):
        new_C = new_C_calc(matM,matK,dx,dt,D,new_C)
        print(new_C.round(2))
        plt.plot(new_C)
        # list_new_C.append(new_C)

m,k,c = form_MK_small()
matM = mat_large(m,rc)
matK = mat_large(k,rc)
C = c_large(c,rc)

# print(matM)
# print(matK)

list_new_C = new_C_repeat()
plt.show()
# print(list_new_C)