# import libs
import numpy as np
import helper_hpc as helper
import os
import torch
from functools import partial
from net import Net

print('file not complete, do not use yet')
exit()

def run():
    torch.multiprocessing.freeze_support()
    
    experiment_name = 'hpc randimg diversity'
    new_experiment_name = 'hpc randimg diversity reinit range'
    name = 'fitness'
    num_runs = 5
    final_gen_i = 49

    # get filters for network
    filename = 'output/' + experiment_name + '/solutions_over_time_{}.npy'.format(name)
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    stored_filters = np.load(filename)
    np.load = np_load_old

    helper.run(seed=False)

    # reinitialize weights other than first layer
    n = Net(diversity='absolute')
    for run_num in range(num_runs):
        for j in range(len(n.conv_layers)):
            stored_filters[run_num][final_gen_i][j] = n.conv_layers[j].weight.data.cpu().numpy()

            print(max(abs(stored_filters[run_num][final_gen_i][j].flatten())))
    print(stored_filters[0].shape, stored_filters[0][final_gen_i].shape, stored_filters[0][final_gen_i][1].shape)

    new_filename = filename.replace(experiment_name, new_experiment_name)
    
    if not os.path.isdir('output/' + new_experiment_name):
        os.mkdir('output/' + new_experiment_name)

    # save to new output file
    with open(new_filename, 'wb') as f:
        np.save(f, stored_filters)


if __name__ == '__main__':
    run()