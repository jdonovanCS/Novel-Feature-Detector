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
parser.add_argument('--trained_models', nargs='+', help='paths to checkpoints of trained models', type=str)
args = parser.parse_args()

def run():
    
    helper.run(seed=False)
    all_filters = {}
    trained_model_filters = {}
    pattern = r'[0-9]'
    for filename in args.filenames:
        # get filters from numpy file
        np_load_old = partial(np.load)
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        stored_filters = np.load(filename)
        np.load = np_load_old
        print(stored_filters.shape)
        if 'random' in filename or 'mutate-only' in filename or 'orthogonal' in filename or 'xavier' in filename:
            all_filters[filename.split('/solutions_over_time')[0]] = stored_filters[0][0]
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
    
    iterator = iter(data_module_vis.val_dataloader())
    
    # 5 and 19
    
    for i in range(5):
        batch = next(iterator)
        if i == 4 or i == 18:
        
            with torch.no_grad():
                x, y = batch
            print(y)
            print(data_module_vis.dataset_test.classes[y])
            plt.imshow(np.transpose(x[0], (1, 2, 0)))
            plt.show()

    if args.trained_models:
        for filename in args.trained_models:
            net = net.load_from_checkpoint(filename)
            name = filename.split("trained")[-1].split('novel-feature-detectors')[0]
            all_filters[name] = net.get_filters()


    # visualize filters
    first3 = {}
    for k, filters in all_filters.items():
        continue
        print(k)
        # net.set_filters(filters)
        # trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
        # print('diversity score:', net.avg_novelty)
            
        for l, layer in enumerate(filters):
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
    
        # visualize activations
        for f in range(len(filters)):
            filters[f] = torch.tensor(filters[f])
        net.set_filters(copy.deepcopy(filters))
        test = net.get_filters()
        # if sum([torch.equal(test[x], filters[x]) for x in range(len(test))]) == len(test):
        #     print('Matches')
        # trainer.validate(net, dataloaders=data_module_eval.val_dataloader(), verbose=False)
        # print('diversity score:', net.avg_novelty)
    
    # fig, ax = plt.subplots(3, len(first3.keys()), figsize=(len(first3.keys()),3))
    # print(len(first3.keys()))
    # for i, k in enumerate(first3.keys()):
    #     print(len(first3[k]))
    #     for j, vals in enumerate(first3[k]):
    #         ax[j, i].imshow(vals, cmap="gray", vmin=0, vmax=255, interpolation='none')
    #         if j == 0:
    #             ax[j,i].set_title(k)
    # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # plt.tight_layout()
    # plt.show()

    # first3={}
    # min_for_layer = [np.inf for i in range(6)]
    # max_for_layer = [-np.inf for i in range(6)]
    # min_for_layer_for_exp = {k: [np.inf for i in range(6)] for k, filters in all_filters.items()}
    # max_for_layer_for_exp = {k: [-np.inf for i in range(6)] for k, filters in all_filters.items()}
    
    
    for k, filters in all_filters.items():
        continue
        print(k)

        if args.network.lower() == 'vgg16':
            net = BigNet(num_classes=len(classnames), classnames=classnames, diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
        else:
            net = Net(num_classes=len(classnames), classnames=classnames, diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
        net.set_filters(copy.deepcopy(filters))
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
            if index > 0:
                continue
            first3[k+str(index)] = []
            if type(activations[layer][0]) != type(np.zeros((1))):
                activations[layer][0] = activations[layer][0].detach().cpu().numpy()
            for image in range (len(activations[layer][0])): # images in batch
                for channel in range(len(activations[layer][0][image])): # channels in activation
                    values = np.array(activations[layer][0][image][channel]).astype(np.int64)
                    if np.min(values) < min_for_layer[index]: min_for_layer[index] = np.min(values)
                    if np.max(values) > max_for_layer[index]: max_for_layer[index] = np.max(values)
                    if np.min(values) < min_for_layer_for_exp[k][index]: min_for_layer_for_exp[k][index] = np.min(values)
                    if np.max(values) > max_for_layer_for_exp[k][index]: max_for_layer_for_exp[k][index] = np.max(values)
                    if image==0 and channel < 3:
                        first3[k+str(index)].append(values[::])
                    rows = cols = int(np.ceil(np.sqrt(len(activations[layer][0][image]))))
                    plt.subplot(rows, cols, channel+1)
                    # plt.imshow(values, vmin=np.min(values), vmax=np.max(values), interpolation='none')
                    plt.imshow(values, vmin=-19, vmax=19, interpolation='none')
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.show()
    # print(min_for_layer_for_exp, max_for_layer_for_exp, min_for_layer, max_for_layer)
    
    # fig, ax = plt.subplots(3, len(first3.keys()), figsize=(len(first3.keys()),3))
    # print(len(first3.keys()))
    # for i, k in enumerate(first3.keys()):
    #     print(len(first3[k]))
    #     for j, vals in enumerate(first3[k]):
    #         ax[j, i].imshow(vals, cmap='gray', vmin=-19, vmax=16, interpolation='none')
    #         if j == 0:
    #             ax[j,i].set_title(k)
    # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # plt.tight_layout()
    # plt.show()


    # weight dist
    # load filters from trained models?
    from scipy.interpolate import UnivariateSpline
    from scipy.stats.kde import gaussian_kde 
    for k, filters in all_filters.items():
        # continue
        print(k)
        filters = [filters]
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
                count, bins_count = np.histogram(filters_local[layer].flatten(), bins=num_bins_, normed=True)
                
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
                
            pdf = pdf / len(filters_local[0])
            mean = mean / len(filters_local[0])
            std = std / len(filters_local[0])
            
            # ADDITIONAL ATTEMPTS AND INFO --------------------------------
            # maximum = [max(stored_filters[0][0][l]) for l in range(len(stored_filters[0][0]))]
            # minimum = [min(stored_filters[0][0][l]) for l in range(len(stored_filters[0][0]))]
            # plotting PDF and CDF
            # Attempt at smoothing
            # bins_count = bins_count[:-1] + (bins_count[1] - bins_count[0])/2   # convert bin edges to centers
            # f = UnivariateSpline(bins_count, count, s=100)
            # plt.plot(bins_count, f(bins_count), color="blue", label="PDF_{}".format(layer))

            # Another attempt
            # kde = gaussian_kde( filters[layer].flatten() )
            # # these are the values over wich your kernel will be evaluated
            # dist_space = np.linspace( min(filters[layer].flatten()), max(filters[layer].flatten()), 100 )
            # plt.plot( dist_space, kde(dist_space) )
            # END ----------------------------------------------------------------
            
            # Original plots
            axes[layer].plot(bins_count[1:], pdf, color="blue")
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
    from scipy.interpolate import make_interp_spline, BSpline
    fk = list(all_filters.keys())[0]
    input_data = []
    for layer in range(len(all_filters[fk])):
        continue
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
                count, bins_count = np.histogram(filters_local[layer].flatten(), range=[-1, 1], bins=100, normed=True)
                
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
        continue
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
                count, bins_count = np.histogram(activations_local[layer][0].flatten(), bins=num_bins_, normed=True)
                
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
                count, bins_count = np.histogram(activations_local[layer][0].flatten(), range=[-20, 20], bins=100, normed=True)
                
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