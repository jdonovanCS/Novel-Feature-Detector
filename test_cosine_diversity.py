import numpy as np
from functools import partial
import sys
# sys.path.append("..")
import helper_hpc as helper
import pytorch_lightning as pl



# relative diversity k10 prms
# relative diversity k10 furthest
# relative diversity k10 lmean
# relative diversity k10 prms lmean
# relative diversity k10


e_file = 'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_fitness.npy'
ind_files = ['D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_4GoUc2o6r8mhbwgeJHhmbq.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_2F58aoXfzekUrApkozdLS3.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_MFrRiPopgx2XL8ayNcBimX.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_Ue4JSwBZnSPAqkmvumrvAV.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_M8F2Q3C5rsZy5xW93h4MMn.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_jXYHANxKdYpecsXieRcAzs.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_664cCq9L8wwiS8dGq8Ehz3.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_SZWsfsVoQ2BsfJYUs64oc8.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_i2cQVkkjhBw2m2FD9cpeGh.npy',
             'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_d8XLvqiRaztbiQvorcD2mt.npy']

np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
e_filters = np.load(e_file)
i_filters = [np.load(f) for f in ind_files]
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


print((type(e_filters)))
print((type(e_filters[0])))
print(len(e_filters[0][0]))
print((type(e_filters[0][0][0])))
print((type(e_filters[0][0][0][0])))
print((type(e_filters[0][0][0][0][0])))
print((type(e_filters[0][0][0][0][0][0])))
print((type(e_filters[0][0][0][0][0][0][0])))
print()

print((e_filters.shape))
print((e_filters[0].shape))
print(len(e_filters[0][0]))
print((e_filters[0][0][0].shape))
print((e_filters[0][0][0][0].shape))
print((e_filters[0][0][0][0][0].shape))
print((e_filters[0][0][0][0][0][0].shape))
print((e_filters[0][0][0][0][0][0][0].shape))
print()

import torch
torch.set_printoptions(precision=32)

print((e_filters[0][0][0][0][0]))
print((e_filters[0][0][0][0][1]))
print()

# For each run
for i in range(len(e_filters)):
    # print(helper.cosine_dist(np.array(e_filters[i][49][0][0][0]).flatten(), np.array(e_filters[i][49][0][0][1]).flatten()))
    # print(helper.cosine_dist(np.array(e_filters[i][6][0][0][0]).flatten(), np.array(e_filters[i][6][0][0][1]).flatten()))
    # For each generation
    num_diffs = 0
    for j in range(0, len(e_filters[i]), len(e_filters[i])):
        
        # For each layer
        for k in range(len(e_filters[i][j])):
            if torch.sum(i_filters[i][j][k] == i_filters[i][49][k]) != 0:
                num_diffs += int(torch.sum(i_filters[i][j][k] != i_filters[i][49][k]))/9
                #    - len(i_filters[i][j][k].flatten()))
                # sum_of_ee = torch.sum(torch.eq(e_filters[i][j][k], e_filters[i][4][j]))
                # sum_of_ei = torch.sum(torch.eq(e_filters[i][j][k], i_filters[i][j][k]))
                # sum_of_ii = torch.sum(i_filters[i][j][k] == i_filters[i][49][k])
                # len_of_filters = len(i_filters[i][j][k].flatten())
                # assert(sum_of_ee == len_of_filters)
                # print('run:', i, 'gen:', j, 'layer:', k, sum_of_ii == len_of_filters)
        #       # assert(sum_of_ii == len_of_filters)

        #     # For each filter
        #     # print(diff)
        #     for l in range(len(e_filters[i][j][k])):
                
    #             for m in range(len(e_filters[i][j][k][l])):
    #                 diff=diff+1 if torch.sum(torch.eq(i_filters[i][j][k][l], i_filters[i][49][k][l])) != len(i_filters[i][j][k][l].flatten()) else diff
    #                 count+=1

    #                 for n in range(len(e_filters[i][j][k][l][m])):
    #                     diff2=diff2+1 if torch.sum(torch.eq(i_filters[i][j][k][l][m], i_filters[i][49][k][l][m])) != len(i_filters[i][j][k][l][m].flatten()) else diff2
    #                     count2+=1
    print(num_diffs)


    # # helper.run(seed=False)
    # data_module = helper.get_data_module("random", batch_size=64, workers=0)
    # data_module.prepare_data()
    # data_module.setup()
    # trainer = pl.Trainer(accelerator="auto", limit_val_batches=1)
    # trainer2 = pl.Trainer(accelerator="auto", limit_val_batches=1)
    # classnames = list(data_module.dataset_test.classes)
    # net = helper.Net(num_classes=len(classnames), classnames=classnames, diversity={"type": 'cosine', "pdop": 'mean', "ldop":'w_mean', 'k': -1, 'k_strat': 'closest'})
    # net2 = helper.Net(num_classes=len(classnames), classnames=classnames, diversity={"type": 'cosine', "pdop": 'mean', "ldop":'w_mean', 'k': -1, 'k_strat': 'closest'})
    # net.set_filters(i_filters[i][10])
    # net2.set_filters(i_filters[i][49])
    # trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
    # trainer2.validate(net2, dataloaders=data_module.val_dataloader(), verbose=False)
    # print(net.avg_novelty, net2.avg_novelty)
        


# for i in range(len(ind_files)):
#     for j in range(len(e_filters[j][0])):
#         assert(torch.sum(torch.eq(e_filters[i][49][j], i_filters[i][49][j])) == )
            # print('match: ' + ind_files[i] + '== e_filters[' + str(j) + ']')
                



# print(sys.getsizeof(e_filters))






