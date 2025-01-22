import numpy as np 

a = np.zeros((2,3,4))
b = np.zeros((4,3,2))
res = a@b
print(a@b)

print(res.shape)