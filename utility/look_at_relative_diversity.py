import numpy as np
from functools import partial
from model import Model
import helper_hpc as helper
import torch

# np.load('output/relative diversity/solutions_over_time_current_fitness_ZKss5bBiLDWY9jpM7a4CM9.npy', allow_pickle=True)

np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

a_filters = np.load('output/relative diversity cifar10/solutions_over_time_fitness.npy')
files = ['solutions_over_time_current_fitness_8LkfpNRhBSbVgRwdFELpLq.npy',
         'solutions_over_time_current_fitness_8rjfajhkKBB4jtuJNNz7Z4.npy',
         'solutions_over_time_current_fitness_52vJDsMJqnMNctihASsYok.npy',
         'solutions_over_time_current_fitness_D3tzpuAzgmZzNsZ4rhcrSW.npy',
         'solutions_over_time_current_fitness_EKvkesD8S7quYtmjSiTZqZ.npy',
         'solutions_over_time_current_fitness_fztucUrVdqBK4hHE6KNHmv.npy',
         'solutions_over_time_current_fitness_oYYZA8aPUzJgCiNmx9fypA.npy',
         'solutions_over_time_current_fitness_PV3FPBE6qnLqTZf7sMiTSh.npy',
         'solutions_over_time_current_fitness_UdXAi5peqTUhzt6KxZRT98.npy',
         'solutions_over_time_current_fitness_XmhMVexm5wxdFjxtJk7oBn.npy']

# for f in files:
#     b_filters = np.load('output/relative diversity cifar10/' + f)
#     for run in range(len(a_filters)):
#         if sum([torch.equal(a_filters[run][49][x], b_filters[49][x]) for x in range(len(a_filters[0][49]))]) == len(a_filters[0][49]):
#             print('Match', f)
# exit()
# print(a_filters.shape)
# print(b_filters.shape)
# print(type(a_filters[0]))
# print(type(b_filters[0]))
# print(type(a_filters[0][0]))
# print(type(b_filters[0][0]))
# exit()

# files = ['solutions_over_time_current_fitness_ZKss5bBiLDWY9jpM7a4CM9.npy',
#          'solutions_over_time_current_fitness_kjcAnjdVEDn9dnrdJ72hij.npy', 
#          'solutions_over_time_current_fitness_RgyXTmEgeSVqLwEztqRnh5.npy', 
#          'solutions_over_time_current_fitness_GQyurk8qoEMVZuzvoXQHqW.npy', 
#          'solutions_over_time_current_fitness_g8fpEE92wUbNLQPGgMyM8s.npy', 
#          'solutions_over_time_current_fitness_cDaz9sF9y48R7Z3NcXsbML.npy', 
#          'solutions_over_time_current_fitness_BA6PXGKcPJa7rc3L9brnEZ.npy', 
#          'solutions_over_time_current_fitness_9aSxCW8pY4YMps5x9umof5.npy', 
#          'solutions_over_time_current_fitness_8F3DdszaNFUPsFBkisLbB3.npy', 
#          'solutions_over_time_current_fitness_2P3mxzFzK4hiq4nGYu24x3.npy']

k = 0
solution_results = {}
solution_results['fitness'] = np.array([None for j in range(10)])
for filename in files:
    print(filename)
    stored_filters = np.load('output/relative diversity cifar10/{}'.format(filename))

    solution_results['fitness'][k] = [list(s) for s in stored_filters]
    k+=1
    for i in range(len(stored_filters)):
        for j in range(len(stored_filters[0])):
            if stored_filters[i][j].shape != stored_filters[0][j].shape:
                print('Failed', i, j, stored_filters[0][0].shape, stored_filters[i][j].shape)

with open('output/transition/relative diversity cifar10/solutions_over_time_fitness.npy', 'wb') as f:
    res = np.array([np.array(v) for v in list(solution_results.values())[0]])
    # for i in range(len(res)):
        # print(type(res[i]))
        # print(res[i].shape)
        # for j in range(len(list(res[i]))):
        #     res[i][j] = list(res[i][j])
    np.save(f, res)

# for k,v in solution_results.items():
#     with open('output/relative diversity scaled/solutions_over_time_fitness.npy', 'wb') as f:
#         np.save(f, [v[-10]])

# saved_filters = np.load('output/relative diversity scaled/solutions_over_time_fitness.npy')
np.load = np_load_old
# print(saved_filters.shape)
    
# for i in range(1, 10):
#     for k, v in solution_results.items():
#         print(i)
#         print(v[-10+i].shape)
#         helper.save_npy('output/relative diversity scaled/solutions_over_time_fitness.npy', v, index=-10+i)

    # print('gens', len(stored_filters))
    # for i in range(len(stored_filters)):
    #     print('layers', len(stored_filters[i]))
        # for j in range(len(stored_filters[i])):
        #     print('layers', len(stored_filters[i][j]))

