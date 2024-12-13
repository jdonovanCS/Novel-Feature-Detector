import numpy as np
import sys
from operator import itemgetter

data = np.load("D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/testing mutate only 750 weighted with indices logging/mutated_filter_indices_mutate-only.npy")
np.set_printoptions(threshold=sys.maxsize)
# np.array(data, dtype=[('layer', int), ('dim1', int), ('dim2', int)])
# data.sort()
print(data.shape)
# for i in range(len(data)):
#     data[i] = sorted(data[i], key=lambda t: t[0])
for i in range(0, 1): #len(data)):
    data[i] = data[i][np.lexsort((data[i][:, 2], data[i][:, 1], data[i][:, 0]))]
    print(data[i][0][0])
    print(data[i])