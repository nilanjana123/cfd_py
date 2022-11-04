import numpy as np 
import matplotlib.pyplot as plt  

R = 8.315
H_a = 140.2
H_b = 181
m_a = 282.47
m_b = 228.37
T_a = 273.8
T_b = 326.6

x_a = np.arange(0.9,0.99, 0.0002) 
x_b = 1 - x_a 

Tm_a = 1/((1/T_a)- ( ( R * np.log(x_a) ) / ( H_a * m_a )))
Tm_b = 1/((1/T_b)- ( ( R * np.log(x_b) ) / ( H_b * m_b )))

# idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
np.set_printoptions(precision=2)
print(Tm_a)
print(Tm_b)

print(Tm_a - Tm_b)
# print(x_b)

plt.plot(x_a, Tm_a) 
plt.plot(x_a, Tm_b) 
plt.show() 