import numpy as np
import math

a = np.mat(np.ones([4,3]))
b = np.mat([[1,2],
            [10,20],
            [100, 200]])

c = np.zeros([4,1])
d = np.ones([10])
# e = c.dot(d)

print(d.reshape([-1,1]).shape)