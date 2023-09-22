# Visualize activations and filters for evolved and random filters

import matplotlib.pyplot as plt
import argparse
import helper_hpc as helper
import wandb
import pickle
from net import Net
from big_net import Net as BigNet
import numpy as np
import os
import time
import torch
from functools import partial
import pytorch_lightning as pl
import re
import copy

# arguments
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--filenames', help="enter filenames", nargs='+', type=str)
parser.add_argument('--dataset_eval', help='which dataset do we want to push through network to evaluate / verify score', default='random')
parser.add_argument('--dataset_vis', help='dataset used to visualize filters when pushed through network', default='cifar-100')
parser.add_argument('--network', help='which network do we want to use for the evaluation', default='conv-6')
args = parser.parse_args()

def run():
    
    helper.run(seed=False)
    all_filters = {}
    pattern = r'[0-9]'
    for filename in args.filenames:
        # get filters from numpy file
        np_load_old = partial(np.load)
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        stored_filters = np.load(filename)
        np.load = np_load_old
        print(stored_filters.shape)
        if 'random' in filename:
            all_filters[filename.split('/solutions_over_time')[0]] = stored_filters[1][0]
        else:
            name = filename.split('/solutions_over_time')[0]
            # for i in range(len(stored_filters[0])):
            #     if i!= len(stored_filters[0])-1 and sum([torch.equal(stored_filters[0][i][x], stored_filters[0][len(stored_filters[0])-1][x]) for x in range(len(stored_filters[0][i]))]) == len(stored_filters[0][i]):
            #         continue
            # all_filters[name+str(9)] = stored_filters[0][9]
            all_filters[name+str(49)] = stored_filters[0][49]
            # if i == len(stored_filters[0])-1:
            with open(name +'/fitness_over_time.txt') as f:
                print(f.read())
        

    # get data to push into network
    data_module_eval = helper.get_data_module(args.dataset_eval, batch_size=64, workers=2)
    data_module_eval.prepare_data()
    data_module_eval.setup()

    data_module_vis = helper.get_data_module(args.dataset_vis, batch_size=1, workers=2)
    data_module_vis.prepare_data()
    data_module_vis.setup()
    classnames = list(data_module_vis.dataset_test.classes)

    if args.network.lower() == 'vgg16':
        net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
    else:
        net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
    trainer = pl.Trainer(accelerator="auto", limit_val_batches=1)
    
    # visualize filters
    for k, filters in all_filters.items():
        print(k)
        # net.set_filters(filters)
        # trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
        # print('diversity score:', net.avg_novelty)
            
        for layer in filters:
            if layer.shape[1] > 3:
                continue
            print(layer.shape)
            plt.figure()
            for i in range(len(layer)):
                for j in range(len(layer[i])):
                    values = np.array(255*((layer[i][j] + 1) /2)).astype(np.int64)
                    rows = len(layer[i])
                    cols = len(layer)
                    plt.subplot(rows, cols, i*len(layer[0]) + (j+1))
                    plt.imshow(values, cmap="gray", vmin = 0, vmax = 255,interpolation='none')
                # values = np.array(255*((layer[i] + 1) /2)).astype(np.int64)
                # if len(np.where(values > 255)) > 0:
                #     print(values)
                #     print(layer[i])
                # rows = cols = int(np.ceil(np.sqrt(len(layer))))
                # plt.subplot(rows, cols, i+1)
                # plt.imshow(values.transpose(1, 2, 0), vmin = 0, vmax = 255,interpolation='none')
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.show()
    
    
        # visualize activations
        
        net.set_filters(copy.deepcopy(filters))
        test = net.get_filters()
        if sum([torch.equal(test[x], filters[x]) for x in range(len(test))]) == len(test):
            print('Matches')
        # trainer.validate(net, dataloaders=data_module_eval.val_dataloader(), verbose=False)
        # print('diversity score:', net.avg_novelty)

    for k, filters in all_filters.items():

        if args.network.lower() == 'vgg16':
            net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
        else:
            net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
        net.set_filters(copy.deepcopy(filters))
        test = net.get_filters()
        if sum([torch.equal(test[x], filters[x]) for x in range(len(test))]) == len(test):
            print('Matches')
        
        batch = next(iter(data_module_vis.val_dataloader()))
        
        with torch.no_grad():
            x, y = batch
            logits = net.forward(x, get_activations=True)
            
            # torch.save(x, 'tensor.pt')
            # x = torch.load('tensor.pt')
            net.forward(x, get_activations=True)
            activations = net.get_activations()
        l = []
        for layer in activations: #layers
            plt.figure()
            print(len(activations[layer]))
            if type(activations[layer][0]) != type(np.zeros((1))):
                activations[layer][0] = activations[layer][0].detach().cpu().numpy()
            print(len(activations[layer][0]))
            print(len(activations[layer][0][0]))
            print(len(activations[layer][0][0][0]))
            for image in range (len(activations[layer][0])): # images in batch
                for channel in range(len(activations[layer][0][image])): # channels in activation
                    values = np.array(activations[layer][0][image][channel]).astype(np.int64)
                    if len(np.where(values > 255)) > 0:
                        pass
                        # print(values)
                        # print(activations[layer][0][image])
                    rows = cols = int(np.ceil(np.sqrt(len(activations[layer][0][image]))))
                    plt.subplot(rows, cols, channel+1)
                    plt.imshow(values, interpolation='none')
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.show()
            break
    exit()
    


if __name__ == '__main__':
    run()