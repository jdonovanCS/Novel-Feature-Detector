import numpy as np
import helper_hpc as helper
import time
import scipy.spatial.distance as dst
import numba
from numpy.linalg import norm
from numpy import dot

@numba.njit(parallel=True)
def cosine_dist(u, v):
    uv=0
    uu=0
    vv=0
    for i in range(u.shape[0]):
        uv+=u[i]*v[i]
        uu+=u[i]*u[i]
        vv+=v[i]*v[i]
    cos_theta=1
    if uu!=0 and vv!=0:
        cos_theta=uv/np.sqrt(uu*vv)
    return 1-cos_theta

answers = [[], [], []]
for i in range(3):
    start = time.time()
    for j in range(100000):
        one = np.random.rand(32*32)
        two = np.random.rand(32*32)
        if i == 0:
            sim = dot(one, two)/(norm(one)*norm(two))
            dist = 1-sim
        if i == 1:
            dist = dst.cosine(one, two)
        if i == 2:
            dist = cosine_dist(one, two)
        answers[i].append(dist)
    end=time.time()
    print("Time for {} = {}".format(i, end-start))  
print(np.array(answers).shape)  


