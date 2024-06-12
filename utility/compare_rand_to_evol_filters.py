import numpy as np
from functools import partial
import sys

# relative diversity k10 prms
# relative diversity k10 furthest
# relative diversity k10 lmean
# relative diversity k10 prms lmean
# relative diversity k10


r_file = 'output/random uniform/solutions_over_time_uniform.npy'
e_file = 'output/relative diversity k10/solutions_over_time_fitness.npy'
np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
e_filters = np.load(e_file)
r_filters = np.load(r_file)
np.load = np_load_old

print((e_filters.dtype))
print((e_filters[0].dtype))
print(len(e_filters[0][0]))
print((e_filters[0][0][0].dtype))
print((e_filters[0][0][0][0].dtype))
print((e_filters[0][0][0][0][0].dtype))
print((e_filters[0][0][0][0][0][0].dtype))
print((e_filters[0][0][0][0][0][0][0].dtype))
print()

print((r_filters.dtype))
print((r_filters[0].dtype))
print(len(r_filters[0][0]))
print((r_filters[0][0][0].dtype))
print((r_filters[0][0][0][0].dtype))
print((r_filters[0][0][0][0][0].dtype))
print((r_filters[0][0][0][0][0][0].dtype))
print((r_filters[0][0][0][0][0][0][0].dtype))
print()
print('-------------------')


print((type(e_filters)))
print((type(e_filters[0])))
print(len(e_filters[0][0]))
print((type(e_filters[0][0][0])))
print((type(e_filters[0][0][0][0])))
print((type(e_filters[0][0][0][0][0])))
print((type(e_filters[0][0][0][0][0][0])))
print((type(e_filters[0][0][0][0][0][0][0])))
print()

print((type(r_filters)))
print((type(r_filters[0])))
print(len(r_filters[0][0]))
print((type(r_filters[0][0][0])))
print((type(r_filters[0][0][0][0])))
print((type(r_filters[0][0][0][0][0])))
print((type(r_filters[0][0][0][0][0][0])))
print((type(r_filters[0][0][0][0][0][0][0])))
print()
print('-------------------')


print((e_filters.shape))
print((e_filters[0].shape))
print(len(e_filters[0][0]))
print((e_filters[0][0][0].shape))
print((e_filters[0][0][0][0].shape))
print((e_filters[0][0][0][0][0].shape))
print((e_filters[0][0][0][0][0][0].shape))
print((e_filters[0][0][0][0][0][0][0].shape))
print()

print((r_filters.shape))
print((r_filters[0].shape))
print(len(r_filters[0][0]))
print((r_filters[0][0][0].shape))
print((r_filters[0][0][0][0].shape))
print((r_filters[0][0][0][0][0].shape))
print((r_filters[0][0][0][0][0][0].shape))
print((r_filters[0][0][0][0][0][0][0].shape))
print()
print('-------------------')

import torch
torch.set_printoptions(precision=32)

print((e_filters[0][0][0][0][0]))
print()

print((r_filters[0][0][0][0][0]))
print()
print('-------------------')


print(sys.getsizeof(e_filters))
print(sys.getsizeof(r_filters))

with open(r_file, 'wb') as f:
    np.save(f, r_filters)

for a in range(len(r_filters)):
    for b in range(len(r_filters[0])):

        for i in range(len(r_filters[0][0])):
            if type(r_filters[a][b][i]) != torch.Tensor:
                print(type(r_filters[a][b][i]))
            if r_filters[a][b][i].dtype != torch.float32:
                print(r_filters[a][b][i].dtype)




