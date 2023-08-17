import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import pickle
import os
import argparse
import random
from functools import partial

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--dataset', help='which dataset should be used for training metric, choices are: cifar-10, cifar-100', default='cifar-100')
parser.add_argument('--fixed_conv', help='Should the convolutional layers stay fixed, or alternatively be trained', action='store_true')
parser.add_argument('--training_interval', help='How often should the network be trained. Values should be supplied as a fraction and will relate to the generations from evolution' +
'For example if 1 is given the filters generated from the final generation of evolution will be the only ones trained. If 0.5 is given then the halfway point of evolutionary generations and the final generation will be trained. ' +
'If 0 is given, the filters from every generation will be trained', type=float, default=1.)
parser.add_argument('--epochs', help="Number of epochos to train for", type=int, default=256)
parser.add_argument('--devices', help='number of gpus to use', default=1, type=int)
parser.add_argument('--local_rank', metavar="N", help='if using ddp and multiple gpus, we only want to collect metrics once, input 0 here if using ddp and multi gpus', default=-1, type=int)

# parser.add_argument('--rand_norm', action='store_true')
parser.add_argument('--gram-schmidt', help='gram-schmidt used to orthonormalize filters', action='store_true')
parser.add_argument('--novelty_interval', help='How often should a novelty score be captured during training?', default=0)
parser.add_argument('--test_accuracy_interval', help='How often should test accuracy be assessed during training?', default=4)
parser.add_argument('--batch_size', help="batch size for training", type=int, default=64)
parser.add_argument('--lr', help='Learning rate for training', default=.001, type=float)

# used to link to evolution
parser.add_argument('--experiment_name', help='experiment name for saving data related to training')
parser.add_argument('--rand_tech', help='which random technique is used to initialize network weights', type=str, default=None)
# don't need any of the below for comparisons since I can link with the above experiment name.

# used to link to autoencoder
parser.add_argument('--ae', help="if pretrained using ae include this tag", action='store_true')

# Options for flexibility
parser.add_argument('--unique_id', help='if a unique id is associated with the file the solution is stored in give it here.', default="", type=str)
parser.add_argument('--skip', default=0, help='skip the first n models to train, used mostly when a run fails partway through', type=int)
parser.add_argument('--stop_after', default=np.inf, help='stop after the first n models', type=int)
parser.add_argument('--num_workers', help='number of workers for training', default=np.inf, type=int)

# Options for measuring diversity over training time
parser.add_argument('--diversity_type', type=str, default='relative', help='Type of diversity metric to use for this experiment (ie. absolute, relative, original etc.)')
parser.add_argument('--pairwise_diversity_op', default='mean', help='the function to use for calculating diversity metric with regard to pairwise comparisons', type=str)   
parser.add_argument('--layerwise_diversity_op', default='w_mean', help='the function to use for calculating diversity metric with regard to layerwise comparisons', type=str)
parser.add_argument('--k', help='If using k-neighbors for metric calculation, how many neighbors', type=int, default=-1)
parser.add_argument('--k_strat', help='If using k-neigbhors for metric, what strategy should be used? (ie. closest, furthest, random, etc.)', type=str, default='closest')   
args = parser.parse_args()

def run():

    torch.multiprocessing.freeze_support()

    stored_filters = {}
    
    experiment_name = args.experiment_name
    training_interval = args.training_interval
    fixed_conv = args.fixed_conv
    if args.rand_tech:
        name = args.rand_tech
    elif not args.rand_tech and not args.gram_schmidt and not args.ae:
        name = 'fitness'
    
    if args.gram_schmidt:
        name = 'gram-schmidt'
    if args.ae:
        name = 'ae_unsup'
        if training_interval < 1:
            print('please enter valid training interval as ae filters are in the shape num_runs, 1, \{filters\}')
            exit()
    if args.unique_id != "":
        name = 'current_' + name + "_" + args.unique_id
    
    filename = ''
    filename = 'output/' + experiment_name + '/solutions_over_time_{}.npy'.format(name)

    # get filters from numpy file
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    stored_filters = np.load(filename)
    np.load = np_load_old

    if args.unique_id != '':
        stored_filters = [stored_filters]

    # if args.random or args.gram_schmidt:
    #     random.shuffle(stored_filters[0])
    #     stored_filters = np.array(stored_filters)
    #     with open(filename, 'wb') as f:
    #         np.save(f, stored_filters)
    
    helper.run(seed=False, rank=args.local_rank if args.devices > 1 else 0)


    # get loader for train and test images and classes
    data_module = helper.get_data_module(args.dataset, args.batch_size, args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    epochs = args.epochs
    
    helper.config['dataset'] = args.dataset.lower()
    helper.config['batch_size'] = args.batch_size
    helper.config['lr'] = args.lr
    helper.config['experiment_name'] = args.experiment_name
    helper.config['evo'] = not args.rand_tech and not args.gram_schmidt and not args.ae
    helper.config['experiment_type'] = 'training'
    helper.config['fixed_conv'] = fixed_conv == True
    helper.config['diversity_type'] = args.diversity_type
    helper.config['ae'] = args.ae
    helper.config['pairwise_diversity_op'] = args.pairwise_diversity_op
    helper.config['layerwise_diversity_op'] = args.layerwise_diversity_op
    helper.config['k'] = args.k
    helper.config['k_strat'] = args.k_strat
    helper.update_config()
    
    
    # run training and evaluation and record metrics in above variables
    # for each type of evolution ran
    inner_skip = 0
    skip = 0
    if int(training_interval) == 1:
        skip = args.skip
    else:
        inner_skip = args.skip
    
    for run_num in range(int(skip), len(stored_filters)):
        # run_num = np.where(stored_filters == filters_list)[0][0]
        # for each generation train the solution output at that generation
        for i in range(int(inner_skip), len(stored_filters[run_num]), int(1/training_interval)*len(stored_filters[run_num])):
            # if we only want to train the solution from the final generation, continue
            # if (training_interval != 0 and i*1.0 not in [(len(stored_filters[run_num])/(1/training_interval)*j)-1 for j in range(1, min(args.stop_after, int(1/training_interval)+1))]) or (training_interval==0 and i not in range(skip, args.stop_after)):
            #     continue
            scaled = False
            if len(stored_filters[run_num][i]) > 6:
                scaled = True
            if args.diversity_type == "None":
                diversity = None
            else:
                diversity = {'type': args.diversity_type, 'pdop': args.pairwise_diversity_op, 'ldop': args.layerwise_diversity_op, 'k': args.k, 'k_strat': args.k_strat}

            # else train the network and collect the metrics
            helper.config['generation'] = i if (not args.rand_tech and not args.gram_schmidt) else None
            helper.update_config()
            save_path = "trained_models/trained/conv{}_e{}_n{}_r{}_g{}.pth".format(not fixed_conv, experiment_name, name, run_num, i)
            print('Training and Evaluating: {} Gen: {} Run: {}'.format(name, i, run_num))
            record_progress = helper.train_network(data_module=data_module, filters=stored_filters[run_num][i], epochs=epochs, lr=args.lr, save_path=save_path, fixed_conv=fixed_conv, novelty_interval=int(args.novelty_interval), val_interval=int(args.test_accuracy_interval), diversity=diversity, scaled=scaled, devices=args.devices)
            helper.run(seed=False)
            helper.config['dataset'] = args.dataset.lower()
            helper.config['batch_size'] = args.batch_size
            helper.config['lr'] = args.lr
            helper.config['experiment_name'] = args.experiment_name
            helper.config['evo'] = not args.rand_tech and not args.gram_schmidt and not args.ae
            helper.config['experiment_type'] = 'training'
            helper.config['fixed_conv'] = fixed_conv == True
            helper.config['diversity_type'] = args.diversity_type
            helper.config['ae'] = args.ae
            helper.config['pairwise_diversity_op'] = args.pairwise_diversity_op
            helper.config['layerwise_diversity_op'] = args.layerwise_diversity_op
            helper.config['k'] = args.k
            helper.config['k_strat'] = args.k_strat
            helper.update_config()
            # for c in classlist:
            #     classwise_accuracy_record[run_num][i][np.where(classlist==c)[0][0]] = record_accuracy[c]

    # with open('output/' + experiment_name + '/classwise_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(classwise_accuracy_record,f)

if __name__ == '__main__':
    run()
