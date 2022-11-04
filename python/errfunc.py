import math 
import matplotlib.pyplot as plt

# import test.text

# f = open('test.txt', 'r+')
# content = f.read()
# print(content)

cons = 2*math.sqrt(0.03*50)
x = 0.5
c_exact = []
while(x!=1):
    c = 100 * math.erf(x/cons)
    x += 0.01
    c_exact.append(c)

plt.plot(c_exact)
