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
from scipy import stats
import seaborn as sns

# arguments
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--filenames', help="enter filenames", nargs='+', type=str)
parser.add_argument('--dataset_eval', help='which dataset do we want to push through network to evaluate / verify score', default='random')
parser.add_argument('--dataset_vis', help='dataset used to visualize filters when pushed through network', default='cifar-100')
parser.add_argument('--network', help='which network do we want to use for the evaluation', default='conv-6')
parser.add_argument('--trained_models', nargs='+', help='paths to checkpoints of trained models', type=str)
args = parser.parse_args()

def run():
    
    helper.run(seed=False)
    global all_filters
    all_filters = {}
    trained_model_filters = {}
    pattern = r'[0-9]'
    import_filters()
    setup_datamodules()
    test_datamodules()
    visualize_filters()
    verify_activations()
    visualize_activations()
    visualize_weight_dist()
    visualize_weight_dist_only_mutated()
    visualize_weight_dist_only_nonmutated()

    all_filters={}
    import_trained_filters()
    visualize_weight_dist()
    indices = get_mutated_filter_indices_from_file()
    visualize_weight_dist_only_mutated(indices)
    visualize_weight_dist_only_nonmutated(indices)
    visualize_weight_dist_only_mutated()
    visualize_weight_dist_only_nonmutated()

    
        


def import_filters():
    for filename in args.filenames:
        # get filters from numpy file
        np_load_old = partial(np.load)
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        stored_filters = np.load(filename)
        np.load = np_load_old
        print(stored_filters.shape)
        if 'random' in filename or 'mutate-only' in filename or 'orthogonal' in filename or 'xavier' in filename:
            all_filters[filename.split('solutions_over_time')[0].split('output')[1].replace('\\', '')] = [s[0] for s in stored_filters][0:10]
        elif 'ae_unsup' in filename:
            all_filters[filename.split('/solutions_over_time')[0]] = stored_filters[0][0]
        else:
            name = filename.split('/solutions_over_time')[0]
            # for i in range(len(stored_filters[0])):
            #     if i!= len(stored_filters[0])-1 and sum([torch.equal(stored_filters[0][i][x], stored_filters[0][len(stored_filters[0])-1][x]) for x in range(len(stored_filters[0][i]))]) == len(stored_filters[0][i]):
            #         continue
            # all_filters[name+str(9)] = stored_filters[0][9]
            all_filters[name+str(49)] = stored_filters[0][49]
            # if i == len(stored_filters[0])-1:
            if os.path.isfile(name+'/fitness_over_time.txt'):
                with open(name +'/fitness_over_time.txt') as f:
                    print(f.read())


# get data to push into network
def setup_datamodules():
    global data_module_eval
    data_module_eval = helper.get_data_module(args.dataset_eval, batch_size=64, workers=2)
    data_module_eval.prepare_data()
    data_module_eval.setup()

    global data_module_vis
    data_module_vis = helper.get_data_module(args.dataset_vis, batch_size=1, workers=2)
    data_module_vis.prepare_data()
    data_module_vis.setup()
    
    global classnames
    classnames = list(data_module_vis.dataset_test.classes)

    global net
    net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'}) 
    if args.network.lower() == 'vgg16':
        net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
        
    trainer = pl.Trainer(accelerator="auto", limit_val_batches=1)

    global batch
    global iterator
    iterator = iter(data_module_vis.val_dataloader())
    batch = next(iterator)

# 5 and 19

def test_datamodules():
    for i in range(4):
        batch = next(iterator)
        if i == 3 or i == 17:
        
            with torch.no_grad():
                x, y = batch
            print(y)
            print(data_module_vis.dataset_test.classes[y])
            plt.imshow(np.transpose(x[0], (1, 2, 0)))
            plt.show()

def import_trained_filters():
    if args.trained_models:
        for filename in args.trained_models:
            
            net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'}) 
            if args.network.lower() == 'vgg16':
                net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
        
            net = net.load_from_checkpoint(filename)
            name = filename.split("trained")[-1].split('novel-feature-detectors')[0]
            all_filters[name] = net.get_filters()


# visualize filters
def visualize_filters():
    first3 = {}
    for k, filters in all_filters.items():
        print(k)
        for run_num in range(len(filters)):
        # net.set_filters(filters)
        # trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
        # print('diversity score:', net.avg_novelty)
            
            for l, layer in enumerate(filters[run_num]):
                if layer.shape[1] > 3:
                    continue
                first3[k+str(l)] = []
                print(layer.shape)
                plt.figure(figsize=(36,3))
                for i in range(len(layer)):
                    for j in range(len(layer[i])):
                        values = np.array(255*((layer[i][j] + 1) /2)).astype(np.int64)
                        rows = len(layer[i])
                        cols = len(layer)
                        plt.subplot(rows, cols, i*len(layer[0]) + (j+1))
                        plt.imshow(values, cmap="gray", vmin = 0, vmax = 255,interpolation='none')
                        if i<3 and j == 0:
                            first3[k+str(l)].append(values[::])
                        # plt.imshow(values, cmap="gray",interpolation='none')
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.tight_layout()
                plt.show()
            break

def verify_activations():
    for k, filters in all_filters.items():
        
        print(k)

        for run_num in range(len(filters)):
            if args.network.lower() == 'vgg16':
                net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
            else:
                net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
            # net.set_filters(filters)
            # trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
            # print('diversity score:', net.avg_novelty)
            for f in range(len(filters[run_num])):
                filters[run_num][f] = torch.tensor(filters[run_num][f])
            net.set_filters(copy.deepcopy(filters[run_num]))
            test = net.get_filters()
            # if sum([torch.equal(test[x], filters[x]) for x in range(len(test))]) == len(test):
            #     print('Matches')
            # trainer.validate(net, dataloaders=data_module_eval.val_dataloader(), verbose=False)
            # print('diversity score:', net.avg_novelty)
            break

    # visualize activations
def visualize_activations():
    first3 = {}
    
    for k, filters in all_filters.items():
        print(k)

        for run_num in range(len(filters)):
            if args.network.lower() == 'vgg16':
                net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
            else:
                net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
            net.set_filters(copy.deepcopy(filters[run_num]))
            test = net.get_filters()
            # if sum([torch.equal(test[x], filters[x]) for x in range(len(test))]) == len(test):
            #     print('Matches')
            
            # batch = next(iter(data_module_vis.val_dataloader()))
            
            with torch.no_grad():
                x, y = batch
                logits = net.forward(x, get_activations=True)
                
                net.forward(x, get_activations=True)
                activations = net.get_activations()
            l = []

            for index, layer in enumerate(activations): #layers
                # plt.figure()
                # only do for first layer since later layers are so much larger
                if index > 6:
                    continue
                first3[k+str(index)] = []
                if type(activations[layer][0]) != type(np.zeros((1))):
                    activations[layer][0] = activations[layer][0].detach().cpu().numpy()
                for image in range (len(activations[layer][0])): # images in batch
                    for channel in range(len(activations[layer][0][image])): # channels in activation
                        values = np.array(activations[layer][0][image][channel]).astype(np.int64)
                        if image==0 and channel < 3:
                            first3[k+str(index)].append(values[::])
                        rows = cols = int(np.ceil(np.sqrt(len(activations[layer][0][image]))))
                        plt.subplot(rows, cols, channel+1)
                        plt.imshow(values, vmin=np.min(values), vmax=np.max(values), interpolation='none')
                        # plt.imshow(values, vmin=-19, vmax=19, interpolation='none')
                    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                    plt.show()
            break


# weight dist
# load filters from trained models?
def old_visualize_weight_dist(): 
    for k, filters in all_filters.items():
        
        print(k)
        # filters = [filters]
        num_runs = len(filters)
        pdf = 0
        fig, axes = plt.subplots(len(filters[0]), figsize=(25, 15))
        for layer in range(len(filters[0])):
            pdf = mean = std = 0
            # divisor = max(abs(filters[0][layer].flatten()))
            # multiplier = max(abs(filters[0][layer].flatten()))
            # num_bins_ = int(500*multiplier)
            multiplier = len(filters[0][layer].flatten())
            num_bins_ = int(.3*multiplier)
            num_outside = 0
            for run_num in range(num_runs):
                filters_local = filters[run_num]
                
                num_outside += sum(abs(filters_local[layer].flatten()))

                # max(np.abs(filters[layer].flatten()))
                # getting data of the histogram
                count, bins_count = np.histogram(filters_local[layer].flatten(), bins=num_bins_, density=True)
                
                # verify sum to 1
                widths = bins_count[1:] - bins_count[:-1]
                assert sum(count * widths) > .99 and sum(count * widths) < 1.01


                # finding the PDF of the histogram using count values
                pdf += count / sum(count)

                mean += filters_local[layer].flatten().mean()
                std += filters_local[layer].flatten().std()
                
                # using numpy np.cumsum to calculate the CDF
                # We can also find using the PDF values by looping and adding
            cdf = np.cumsum(pdf)

            kde = stats.gaussian_kde(bins_count[1:])
            bins_count_x = np.linspace(bins_count[1:].min(), bins_count[1:].max(), num_bins_)
                
            pdf = pdf / len(filters_local[0])
            mean = mean / len(filters_local[0])
            std = std / len(filters_local[0])
            
            # Original plots
            axes[layer].plot(bins_count[1:], pdf, color="blue")
            axes[layer].plot(bins_count_x, kde(bins_count_x))
            axes[layer].set_title("PDF_{}".format(layer+1))
            # plt.plot(bins_count[1:], cdf, label="CDF")
            # print(mag)
            perc_outside = num_outside/len(filters_local[0][layer].flatten())/(num_runs)
            perc_inside = 1-perc_outside
            # print('percentage of weights outside of the range: -{}, {}: {}'.format(divisor, divisor, perc_outside))
            # print('percentage of weights inside of the range: -{}, {}: {}'.format(divisor, divisor, perc_inside))
            print('mean: {} \t std: {}'.format(mean, std))
        plt.tight_layout()
        plt.show()

# weight dist, but remove filters that haven't been mutated
# load filters from trained models?
def old_visualize_weight_dist_only_mutated(indices=None): 
    for key, filters in all_filters.items():
        mutated_filter_indices = []

        # continue
        print(key)

        # filters = [filters]
        num_runs = len(filters)
        pdf = 0
        fig, axes = plt.subplots(len(filters[0]), figsize=(25, 15))
        # can put if statement here for if indices == None. If not, then just add the ones we know matter.. duh
        for layer in range(0, len(filters[0])):
            pdf = mean = std = 0
            num_bins_ = int(len(filters[0][layer].flatten()) / 1000)
            num_outside = 0
            for run_num in range(num_runs):
                filters_local = filters[run_num]
                
                num_outside += sum(abs(filters_local[layer].flatten()))

                filter_values_local = []
                
                for c, channel in enumerate(filters_local[layer]):
                    for f, filter in enumerate(channel):
                        # could make this if a bit more efficient if I break out the np_equal
                        if ((indices is None) and (torch.sum(torch.abs(filter) > (1/np.sqrt(len(channel))) + .1) > 0)) or ((indices is not None) and (any(np.array_equal([run_num, layer, c, f], row) for row in indices))): 
                                mutated_filter_indices.append((run_num, layer, c, f))
                                for val in filter:
                                    filter_values_local.append(val[0])
                                

                filter_values_local = np.array(filter_values_local)
                if len(filter_values_local) == 0:
                    continue
                # print(filter_values_local)
                count, bins_count = np.histogram(filter_values_local.flatten(), bins=num_bins_, density=True)
                # sns.distplot(filter_values_local.flatten(), bins=num_bins_)
                
                # verify sum to 1
                widths = bins_count[1:] - bins_count[:-1]
                assert sum(count * widths) > .99 and sum(count * widths) < 1.01


                # finding the PDF of the histogram using count values
                pdf += count / sum(count)

                mean += filter_values_local[layer].flatten().mean()
                std += filter_values_local[layer].flatten().std()
                
                # using numpy np.cumsum to calculate the CDF
                # We can also find using the PDF values by looping and adding
            
            if len(filter_values_local) == 0:
                continue
            kde = stats.gaussian_kde(bins_count[1:])
            bins_count_x = np.linspace(bins_count[1:].min(), bins_count[1:].max(), len(pdf))


            cdf = np.cumsum(pdf)
                
            pdf = pdf / len(filters_local[0])
            mean = mean / len(filters_local[0])
            std = std / len(filters_local[0])
            
            # Original plots
            axes[layer].plot(bins_count[1:], pdf, color="blue")
            axes[layer].plot(bins_count_x, kde(pdf))
            axes[layer].set_title("PDF_{}".format(layer+1))
            # plt.plot(bins_count[1:], cdf, label="CDF")
            # print(mag)
            perc_outside = num_outside/len(filters_local[0][layer].flatten())/(num_runs)
            perc_inside = 1-perc_outside
            # print('percentage of weights outside of the range: -{}, {}: {}'.format(divisor, divisor, perc_outside))
            # print('percentage of weights inside of the range: -{}, {}: {}'.format(divisor, divisor, perc_inside))
            print('mean: {} \t std: {}'.format(mean, std))
        plt.tight_layout()
        plt.show()

        if not os.path.isfile('output/' + args.filenames[i].split('solutions_over_time')[0].split('output')[1].replace('\\', '') + '/mutated_filter_indices.npy'):
            with open('output/' + args.filenames[i].split('solutions_over_time')[0].split('output')[1].replace('\\', '') + '/mutated_filter_indices.npy', 'wb') as f:
                np.save(f, mutated_filter_indices)

def get_mutated_filter_indices_from_file():
    for i in range(len(args.filenames)):
        np_load_old = partial(np.load)
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        indices_file = 'output/' + args.filenames[i].split('solutions_over_time')[0].split('output')[1].replace('\\', '') + '/mutated_filter_indices.npy'
        indices = np.load(indices_file)
        np.load = np_load_old
        return indices


def old_visualize_weight_dist_only_nonmutated(indices=None): 
    for key, filters in all_filters.items():

        # continue
        print(key)

        # filters = [filters]
        num_runs = len(filters)
        pdf = 0
        fig, axes = plt.subplots(len(filters[0]), figsize=(25, 15))
        for layer in range(0, len(filters[0])):
            pdf = mean = std = 0
            num_bins_ = int(len(filters[0][layer].flatten()) / 100)
            num_outside = 0
            for run_num in range(0, 1):#num_runs):
                filters_local = filters[run_num]
                
                num_outside += sum(abs(filters_local[layer].flatten()))

                filter_values_local = []

                for c, channel in enumerate(filters_local[layer]):
                    for f, filter in enumerate(channel):
                        if (indices is None and torch.sum(torch.abs(filter) > (1/np.sqrt(len(channel))) + .1) == 0) or (indices is not None and not any(np.array_equal([run_num, layer, c, f], row) for row in indices)):
                            for val in filter:
                                filter_values_local.append(val[0])

                filter_values_local = np.array(filter_values_local)
                if len(filter_values_local) == 0:
                    continue
                # print(filter_values_local)
                count, bins_count = np.histogram(filter_values_local.flatten(), bins=num_bins_, density=True)
                
                # verify sum to 1
                widths = bins_count[1:] - bins_count[:-1]
                assert sum(count * widths) > .99 and sum(count * widths) < 1.01


                # finding the PDF of the histogram using count values
                pdf += count / sum(count)

                mean += filter_values_local[layer].flatten().mean()
                std += filter_values_local[layer].flatten().std()
                
                # using numpy np.cumsum to calculate the CDF
                # We can also find using the PDF values by looping and adding
            
            if len(filter_values_local) == 0:
                continue

            cdf = np.cumsum(pdf)
            kde = stats.gaussian_kde(bins_count[1:])
            bins_count_x = np.linspace(bins_count[1:].min(), bins_count[1:].max(), num_bins_)
                
            pdf = pdf / len(filters_local[0])
            mean = mean / len(filters_local[0])
            std = std / len(filters_local[0])
            
            # Original plots
            axes[layer].plot(bins_count[1:], pdf, color="blue")
            axes[layer].plot(bins_count_x, kde(bins_count_x))
            axes[layer].set_title("PDF_{}".format(layer+1))
            # plt.plot(bins_count[1:], cdf, label="CDF")
            # print(mag)
            perc_outside = num_outside/len(filters_local[0][layer].flatten())/(num_runs)
            perc_inside = 1-perc_outside
            # print('percentage of weights outside of the range: -{}, {}: {}'.format(divisor, divisor, perc_outside))
            # print('percentage of weights inside of the range: -{}, {}: {}'.format(divisor, divisor, perc_inside))
            print('mean: {} \t std: {}'.format(mean, std))
        plt.tight_layout()
        plt.show()

def visualize_weight_dist():
    for i, (key, filters) in enumerate(all_filters.items()):
        mutated_filter_indices = []

        # filters = [filters]
        # continue
        print(key)

        num_runs = len(filters)
        fig, ax = plt.subplots(figsize=(25, 15))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, ["layer {}".format(i) for i in range(len(filters[0]))], loc='lower right')
        # can put if statement here for if indices == None. If not, then just add the ones we know matter.. duh
        for layer in range(0, len(filters[0])):
            num_bins_ = 100
            filter_values_local = []
            for run_num in range(num_runs):
                filters_local = filters[run_num]
                filter_values_local = np.append(filter_values_local, filters_local[layer].flatten())
            
            if len(filter_values_local) == 0:
                continue

            filter_values_local = np.array(filter_values_local)
            kde = stats.gaussian_kde(filter_values_local)
            filter_values_local_x = np.linspace(filter_values_local.min(), filter_values_local.max(), 1000)
            # Original plots
            # hist, bin_edges = np.histogram(filter_values_local, bins=num_bins_, density=True)
            # axes[layer].plot(bin_edges[1:], hist, alpha=.4)
            ax.plot(filter_values_local_x, kde(filter_values_local_x))
            # axes[layer].set_title("PDF_{}".format(layer+1))
            
            print('mean: {} \t std: {}'.format(filter_values_local.mean(), filter_values_local.std()))
        plt.tight_layout()
        plt.show()

def visualize_weight_dist_only_mutated(indices=None):
    for i, (key, filters) in enumerate(all_filters.items()):
        mutated_filter_indices = []

        # continue
        print(key)

        # filters = [filters]
        num_runs = len(filters)
        fig, ax = plt.subplots(figsize=(25, 15))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, ["layer {}".format(i) for i in range(len(filters[0]))], loc='lower right')
        # can put if statement here for if indices == None. If not, then just add the ones we know matter.. duh
        for layer in range(0, len(filters[0])):
            num_bins_ = 100
            filter_values_local = []
            for run_num in range(num_runs):
                filters_local = filters[run_num]
                mutated_filter_indices.append([])
                
                
                for c, channel in enumerate(filters_local[layer]):
                    for f, filter in enumerate(channel):
                        # could make this if a bit more efficient if I break out the np_equal
                        if ((indices is None) and (torch.sum(torch.abs(filter) > (1/np.sqrt(len(channel))) + .1) > 0)) or ((indices is not None) and (any(np.array_equal([run_num, layer, c, f], row) for row in indices[run_num]))): 
                                mutated_filter_indices[run_num].append((run_num, layer, c, f))
                                for val in filter:
                                    filter_values_local.append(val[0])
            
            if len(filter_values_local) == 0:
                continue

            filter_values_local = np.array(filter_values_local)
            kde = stats.gaussian_kde(filter_values_local)
            filter_values_local_x = np.linspace(filter_values_local.min(), filter_values_local.max(), 100)
            
            # Original plots
            # axes[layer].hist(filter_values_local, bins=num_bins_, alpha=.4, density=True)
            ax.plot(filter_values_local_x, kde(filter_values_local_x))
            # axes[layer].set_title("PDF_{}".format(layer+1))
            
            print('mean: {} \t std: {}'.format(filter_values_local.mean(), filter_values_local.std()))
        plt.tight_layout()
        plt.show()

        
        if not os.path.isfile('output/' + args.filenames[i].split('solutions_over_time')[0].split('output')[1].replace('\\', '') + '/mutated_filter_indices.npy'):
            with open('output/' + args.filenames[i].split('solutions_over_time')[0].split('output')[1].replace('\\', '') + '/mutated_filter_indices.npy', 'wb') as f:
                np.save(f, mutated_filter_indices)

def visualize_weight_dist_only_nonmutated(indices=None):
    for i, (key, filters) in enumerate(all_filters.items()):
        mutated_filter_indices = []

        # continue
        print(key)

        # filters = [filters]
        num_runs = len(filters)
        fig, ax = plt.subplots(figsize=(25, 15))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, ["layer {}".format(i) for i in range(len(filters[0]))], loc='lower right')
        # can put if statement here for if indices == None. If not, then just add the ones we know matter.. duh
        for layer in range(0, len(filters[0])):
            num_bins_ = 100
            filter_values_local = []
            for run_num in range(num_runs):
                filters_local = filters[run_num]
                
                
                for c, channel in enumerate(filters_local[layer]):
                    for f, filter in enumerate(channel):
                        # could make this if a bit more efficient if I break out the np_equal
                        if (indices is None and torch.sum(torch.abs(filter) > (1/np.sqrt(len(channel))) + .1) == 0) or (indices is not None and not any(np.array_equal([run_num, layer, c, f], row) for row in indices[run_num])):
                                mutated_filter_indices.append((run_num, layer, c, f))
                                for val in filter:
                                    filter_values_local.append(val[0])
            
            if len(filter_values_local) == 0:
                continue

            filter_values_local = np.array(filter_values_local)
            kde = stats.gaussian_kde(filter_values_local)
            filter_values_local_x = np.linspace(filter_values_local.min(), filter_values_local.max(), 100)
            
            # Original plots
            # axes[layer].hist(filter_values_local, bins=num_bins_, alpha=.4, density=True)
            ax.plot(filter_values_local_x, kde(filter_values_local_x))
            # axes[layer].set_title("PDF_{}".format(layer+1))
            
            print('mean: {} \t std: {}'.format(filter_values_local.mean(), filter_values_local.std()))
        plt.tight_layout()
        plt.show()

def unknown_code_when_revising():
        from scipy.interpolate import make_interp_spline, BSpline
        fk = list(all_filters.keys())[0]
        input_data = []
        for layer in range(len(all_filters[fk])):
            
            fig, axes = plt.subplots(len(all_filters.keys()), figsize=(25,15))
            i = 0
            for k, filters in all_filters.items():
                input_data.append([])
                filters = [filters]
                num_runs = len(filters)
                pdf = mean = std = 0
                # divisor = max(abs(filters[0][layer].flatten()))
                multiplier = max(abs(filters[0][layer].flatten()))
                num_bins_ = int(500*multiplier)
                num_outside = 0
                for run_num in range(num_runs):
                    filters_local=filters[run_num]
                    num_outside += sum(abs(filters_local[layer].flatten()))

                    # getting data of the histogram
                    count, bins_count = np.histogram(filters_local[layer].flatten(), range=[-1, 1], bins=100, density=True)
                    
                    # verify sum to 1
                    widths = bins_count[1:] - bins_count[:-1]
                    assert sum(count * widths) > .99 and sum(count * widths) < 1.01

                    # finding the PDF of the histogram using count values
                    pdf = count / sum(count)

                    mean = filters_local[layer].flatten().mean()
                    std = filters_local[layer].flatten().std()

                    input_data[i].append(pdf)
                        
                i+=1
            
            plt.rcParams.update({'font.size': 22, "figure.figsize": (7, 6)})
            helper.plot_mean_and_bootstrapped_ci_multiple([np.transpose(x) for x in input_data], '', ['Relative', 'Absolute', 'Cosine', 'Autoencoder', 'Random Normal', 'Random Uniform'], 'Weight', 'Percentage', show=True, alpha=.5, y=bins_count[1:])



        for k, filters in all_filters.items():
            print(k)
            if args.network.lower() == 'vgg16':
                net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
            else:
                net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
            net.set_filters(copy.deepcopy(filters))

            with torch.no_grad():
                x, y = batch
                logits = net.forward(x, get_activations=True)
                
                # torch.save(x, 'tensor.pt')
                # x = torch.load('tensor.pt')
                net.forward(x, get_activations=True)
                activations = net.get_activations()

            activations = [list(activations.values())]
            num_runs = len(activations)
            pdf = 0
            fig, axes = plt.subplots(len(activations[0]), figsize=(25,15))
            for layer in range(len(activations[0])):
                pdf = mean = std = 0
                # divisor = max(abs(filters[0][layer].flatten()))
                multiplier = max(abs(activations[0][layer][0].flatten()))
                num_bins_ = int(500*multiplier)
                num_outside = 0
                for run_num in range(num_runs):
                    activations_local = activations[run_num]
                    
                    num_outside += sum(abs(activations_local[layer][0].flatten()))

                    # max(np.abs(filters[layer].flatten()))
                    # getting data of the histogram
                    count, bins_count = np.histogram(activations_local[layer][0].flatten(), bins=num_bins_, density=True)
                    
                    # verify sum to 1
                    widths = bins_count[1:] - bins_count[:-1]
                    assert sum(count * widths) > .99 and sum(count * widths) < 1.01


                    # finding the PDF of the histogram using count values
                    pdf += count / sum(count)

                    mean += activations_local[layer][0].flatten().mean()
                    std += activations_local[layer][0].flatten().std()
                    
                    # using numpy np.cumsum to calculate the CDF
                    # We can also find using the PDF values by looping and adding
                cdf = np.cumsum(pdf)
                    
                pdf = pdf / len(activations_local[0][0])
                mean = mean / len(activations_local[0][0])
                std = std / len(activations_local[0][0])
                
                # Original plots
                axes[layer].plot(bins_count[1:], pdf, color="blue")
                axes[layer].set_title("PDF_{}".format(layer+1))
                
                perc_outside = num_outside/len(activations_local[layer][0].flatten())/(num_runs)
                perc_inside = 1-perc_outside
                
                print('mean: {} \t std: {}'.format(mean, std))
            plt.show()

        fk = list(all_filters.keys())[0]
        input_data = []
        for layer in range(len(all_filters[fk])):
            # continue
            i = 0
            for k, filters in all_filters.items():
                print(k)
                input_data.append([])
                if args.network.lower() == 'vgg16':
                    net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
                else:
                    net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
                print(type(filters[0]), type(np.array([])))
                if type(filters[0]) == type(np.array([])):
                    filters_copy = copy.deepcopy([torch.from_numpy(f) for f in filters])
                else:
                    filters_copy = copy.deepcopy(filters)
                net.set_filters(filters_copy)

                with torch.no_grad():
                    x, y = batch
                    logits = net.forward(x, get_activations=True)
                    
                    # torch.save(x, 'tensor.pt')
                    # x = torch.load('tensor.pt')
                    net.forward(x, get_activations=True)
                    activations = net.get_activations()

                activations = [list(activations.values())]
                num_runs = len(activations)
                pdf = mean = std = 0
                # divisor = max(abs(filters[0][layer].flatten()))
                # multiplier = max(abs(filters[0][layer].flatten()))
                # num_bins_ = int(500*multiplier)
                num_outside = 0
                for run_num in range(num_runs):
                    activations_local=activations[run_num]
                    num_outside += sum(abs(activations_local[layer][0].flatten()))

                    # getting data of the histogram
                    count, bins_count = np.histogram(activations_local[layer][0].flatten(), range=[-20, 20], bins=100, density=True)
                    
                    # verify sum to 1
                    widths = bins_count[1:] - bins_count[:-1]
                    assert sum(count * widths) > .99 and sum(count * widths) < 1.01

                    # finding the PDF of the histogram using count values
                    pdf = count / sum(count)

                    mean = activations_local[layer][0].flatten().mean()
                    std = activations_local[layer][0].flatten().std()

                    input_data[i].append(pdf)
                        
                i+=1
            
            plt.rcParams.update({'font.size': 22, "figure.figsize": (7, 6)})
            helper.plot_mean_and_bootstrapped_ci_multiple([np.transpose(x) for x in input_data], '', ['Relative', 'Absolute', 'Cosine', 'Autoencoder', 'Random Normal', 'Random Uniform'], 'Feature Map Value', 'Percentage', show=True, alpha=.5, y=bins_count[1:])

    


if __name__ == '__main__':
    run()