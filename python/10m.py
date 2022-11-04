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


# def new_C_calc(matM,matK,dx,dt,D,C):
#     mk = (dx/6)*matM + ((D*dt)/(dx))*matK
#     mk_inv = np.linalg.inv(mk)

#     mc = (dx/6) * np.matmul(matM,C)

#     new_rhs = np.matmul(mk_inv,mc)
#     # print(new_rhs)
#     new_C = np.copy(new_rhs)
#     new_Q = np.array([0]*6,dtype = np.float64)
#     new_C[0] = 0
#     new_C[new_C.size -1] = 0
#     new_Q[0] = new_rhs[0] / (mk_inv[0][0] * dt) #* (-1)
#     new_Q[rc -1] = (new_rhs[rc -1] / (mk_inv[0][rc-1] * dt))*(-1)
#     print(new_C)
#     # print(new_Q)
#     # print(mk_inv[0])
#     # print(mk_inv[0][0],mk_inv[0][rc-1])


#     # print(mat_lhs_inv)
#     # print(mat_rhs)
#     return 0


def new_C_calc(matM,matK,dx,dt,D,C):
    mat_lhs =  (dx/6)*matM + ((D)/(dx))*matK
    mat_lhs_inv = np.linalg.inv(mat_lhs)

    mat_rhs_part = ((dx/(6*dt))*matM - (D/dx)*matK) 

    mat_rhs = np.matmul(mat_rhs_part, C)

    new_C = np.matmul(mat_lhs_inv,mat_rhs)
    # new_C[0] = 0 #
    # new_C[rc-1] = 0 #
   
    # print(mat_lhs_inv)
    # print(mat_rhs_part)
    return(new_C)
    
# list_new_C = np.ndarray()
def new_C_repeat():
    new_C = C
    for i in range(10):
        new_C = new_C_calc(matM,matK,dx,dt,D,new_C)
        print(new_C.round(2))
        # list_new_C.append(new_C)

list_new_C = new_C_repeat()
# print(list_new_C)
# new_C_calc(matM,matK,dx,dt,D,C)
    