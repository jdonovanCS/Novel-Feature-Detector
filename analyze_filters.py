import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
from functools import partial
import pytorch_lightning as pl
import copy
from tqdm import tqdm


# list of filters to load
filename = "output/mutate 1250 weighted .37 .08 .08 .18 .12 .17/solutions_over_time_mutate-only.npy"

# get filters from numpy file
np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
stored_filters = np.load(filename)
np.load = np_load_old

# create pytorch data_module, trainer, and network
data_module = helper.get_data_module('random', batch_size=64, workers=0, shuffle=False)
data_module.prepare_data()
data_module.setup()
trainer = pl.Trainer(accelerator="auto", limit_val_batches=1)
classnames = list(data_module.dataset_test.classes)
net = helper.Net(num_classes=len(classnames), classnames=classnames, diversity={"type": 'relative', "pdop": 'mean', "ldop":'w_mean', 'k': -1, 'k_strat': 'closest'})

# variables to store calculations
diversities = {}
features_list = []
novelties = {}
filters_list = []
filter_novelties = {}

# for each set of filters in list of imported filters (shape=(1, # of layers, # of filters, # of in-channels, size of filters, size of filters))
for i in range(len(stored_filters)):
    for j in range(len(stored_filters[i])):
        net.set_filters(stored_filters[i][j])
        trainer.validate(net, dataloaders=data_module.train_dataloader(), verbose=False)
        features_list.append(copy.deepcopy(net.get_features(numpy=True)))
        diversities[i] = (copy.deepcopy(net.avg_novelty))
        filters_list.append(copy.deepcopy(net.get_filters(numpy=True)))
        

for i in tqdm(range(len(features_list))):
    novelties[i] = (helper.feature_novelty(features_list[i], features_list))
    filter_novelties[i] = (helper.filter_novelty(filters_list[i], filters_list))

print('diversities', {k: v for k, v in sorted(diversities.items(), key=lambda item: item[1])})
print('max diversity', max(diversities.values()))
print('novelties', {k: v for k, v in sorted(novelties.items(), key=lambda item: item[1])})
print('max novelty', max(novelties.values()))
print('filter novelties', {k: v for k, v in sorted(filter_novelties.items(), key=lambda item: item[1])})
print('max filter novelty', max(filter_novelties.values()))

