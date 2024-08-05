import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
from functools import partial
import pytorch_lightning as pl
import copy



filename = "output/random uniform/solutions_over_time_uniform.npy"

# get filters from numpy file
np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
stored_filters = np.load(filename)
np.load = np_load_old

data_module = helper.get_data_module('cifar-10', batch_size=64, workers=0, shuffle=False)
data_module.prepare_data()
data_module.setup()
trainer = pl.Trainer(accelerator="auto", limit_val_batches=1)
classnames = list(data_module.dataset_test.classes)
net = helper.Net(num_classes=len(classnames), classnames=classnames, diversity={"type": 'relative', "pdop": 'mean', "ldop":'w_mean', 'k': -1, 'k_strat': 'closest'})


novelties = []
for i in range(len(stored_filters)):
    for j in range(len(stored_filters[i])):
        net.set_filters(stored_filters[i][j])
        trainer.validate(net, dataloaders=data_module.train_dataloader(), verbose=False)
        novelties.append(copy.deepcopy(net.avg_novelty))

print(novelties)
