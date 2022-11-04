import numpy as np
from sympy import *
# var(x)
na = 3
arr = [1,2,3,4,5,6,7,8,9]
a = np.array(arr, dtype = np.float64).reshape(na,na)
b = np.array([1,2,3], dtype = np.float64)
print(a)
print(b)
rc = 10
mat = np.array([0]*(rc*rc),dtype = np.float64).reshape(rc,rc)
bias = np.array([0]*(rc),dtype = np.float64)
L = np.array([0]*(rc*rc)).reshape(rc,rc)
D = np.array([0]*(rc*rc)).reshape(rc,rc)
U = np.array([0]*(rc*rc)).reshape(rc,rc)

for k in range(0,rc-2,1):
    for i in range(na):
        for j in range(na):
            mat[i+k][j+k] += a[i][j]
        bias[i+k] += b[i]

print(mat)
print(bias)

for i in range(rc):
    for j in range(rc):
        if i>j:
            L[i][j] = mat[i][j]
        if i == j:
            D[i][j] = mat[i][j]
        if i<j:
            U[i][j] = mat[i][j]

# print(L)
# print(D)
# print(U)

xt = []
x = np.array([0]*(rc),dtype = np.float64)
# print(x)
# xt.append(x)
# xt.append(x)
# print(np.array(xt))

# inverse
mat_inv = np.linalg.inv(mat)
# print(mat_inv)
x0 = np.matmul(mat_inv,bias)
# print(x0)
print("gs")
# x = 
for r in range(20):
  for onr in range(na):
    for i in range(3):
        mat1 = mat[i]
        mat1 = np.delete(mat1,i)
        px = np.delete(x,i)
        d = x[i] if x[i]>0 else 1.0
        # px.transpose()
        # print(px)
        # print(mat[i])
        # print(mat1)
        x[i] = (np.matmul(mat1,px) + bias[i])/d
        # print(x)
#   print(x.round(decimals=2))
# print(x)