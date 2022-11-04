import numpy as np

na=2
arrM = [2,1,1,2]
arrK = [0,0,-2,2]
# arrM = [2,1,1,2]
# arrK = [1,-1,-1,1]
m = np.array(arrM, dtype = np.float64).reshape(na,na)
k = np.array(arrK, dtype = np.float64).reshape(na,na)

C = np.array([100,100,100,100],dtype= np.float64)
# C = np.array([100,0,0,0,0,100],dtype= np.float64)

rc = 4
def mat_large(a,rc):
    mat = np.array([0]*(rc*rc),dtype = np.float64).reshape(rc,rc)
    for k in range(0,rc-1,1):
        for i in range(na):
            for j in range(na):
                mat[i+k][j+k] += a[i][j]
    return mat


def sln(matM,matK,dx,dt,D,C):
    pass
matM = mat_large(m,rc)
matK = mat_large(k,rc)
# print(matM)
# print(matK)
dx = 1/3
dt = 0.05
D = 1.64*10**(-4)
print(matM)
print(matK)
# print(np.linalg.det(matM))
# print(np.linalg.det(matK))
'''
def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs =  (dx/6)*matM + ((D)/(dx))*matK
    # mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs_part = ((dx/(6*dt))*matM - (D/dx)*matK) 

    mat_rhs = np.matmul(mat_rhs_part, C)
    
    new_C = np.copy(C)
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
'''

def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs =  (dx/6)*matM + ((D)/(dx))*matK
    mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs_part = ((dx/(6*dt))*matM - (D/dx)*matK) 

    mat_rhs = np.matmul(mat_rhs_part, C)
    new_C = np.matmul(mat_lhs_inv,mat_rhs)
    # new_C[0] = 0 #
    # new_C[rc-1] = 0 #
   
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
# new_C_calc(matM,matK,dx,dt,D,C)
    