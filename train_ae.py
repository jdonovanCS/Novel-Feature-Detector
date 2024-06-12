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
parser.add_argument('--dataset', help='which dataset should be used for training metric, choices are: cifar-10, cifar-100', default='random')
parser.add_argument('--experiment_name', help='experiment name for saving data related to training')
parser.add_argument('--epochs', help="Number of epochos to train for", type=int, default=64)
parser.add_argument('--steps', help='Number of steps to train for', default=10000, type=int)
parser.add_argument('--test_accuracy_interval', help='How often should test accuracy be assessed during training?', default=4)
parser.add_argument('--novelty_interval', help='How often should a novelty score be captured during training?', default=0)
parser.add_argument('--batch_size', help="batch size for training", type=int, default=64)
parser.add_argument('--lr', help='Learning rate for training', default=.001, type=float)
parser.add_argument('--num_batches_for_ae', help='Number of batches used of dataset when calculating diversity of filters', default=np.inf, type=int)
parser.add_argument('--diversity_type', type=str, default='relative', help='Type of diversity metric to use for this experiment (ie. absolute, relative, original etc.)')
parser.add_argument('--pairwise_diversity_op', default='mean', help='the function to use for calculating diversity metric with regard to pairwise comparisons', type=str)   
parser.add_argument('--layerwise_diversity_op', default='w_mean', help='the function to use for calculating diversity metric with regard to layerwise comparisons', type=str)
parser.add_argument('--k', help='If using k-neighbors for metric calculation, how many neighbors', type=int, default=-1)
parser.add_argument('--k_strat', help='If using k-neigbhors for metric, what strategy should be used? (ie. closest, furthest, random, etc.)', type=str, default='closest')
parser.add_argument('--num_workers', help='number of workers for training', default=np.inf, type=int)
parser.add_argument('--num_runs', help='how many ae networks to train', default=5, type=int)
parser.add_argument('--encoded_space_dims', help='final encoding dims flattened', default=256, type=int)
parser.add_argument('--rand_tech', help='random distribution of weights for network', default='uniform', type=str)
args = parser.parse_args()

def run():

    torch.multiprocessing.freeze_support()
    
    experiment_name = args.experiment_name
    
    helper.run(seed=False)


    # get loader for train and test images and classes
    data_module = helper.get_data_module(args.dataset, args.batch_size, args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    epochs = args.epochs
    steps = args.steps
    
    helper.config['dataset'] = args.dataset.lower()
    helper.config['batch_size'] = args.batch_size
    helper.config['lr'] = args.lr
    helper.config['experiment_name'] = args.experiment_name
    helper.config['experiment_type'] = 'ae_unsup_train'
    helper.config['diversity_type'] = args.diversity_type
    helper.config['pairwise_diversity_op'] = args.pairwise_diversity_op
    helper.config['layerwise_diversity_op'] = args.layerwise_diversity_op
    helper.config['k'] = args.k
    helper.config['k_strat'] = args.k_strat
    helper.config['rand_tech'] = args.rand_tech
    helper.update_config()
    
    
    # run training and evaluation and record metrics in above variables
    # for each type of evolution ran
    for run_num in range(args.num_runs):
        # run_num = np.where(stored_filters == filters_list)[0][0]
        # for each generation train the solution output at that generation

        scaled = False
            
        # else train the network and collect the metrics
        save_path = "trained_models/trained/ae_e{}_r{}.pth".format(experiment_name, run_num)
        print('Training and Evaluating Run: {}'.format(run_num))
        record_progress = helper.train_ae_network(data_module=data_module, epochs=epochs, steps=steps, lr=args.lr, encoded_space_dims=args.encoded_space_dims, save_path=save_path, novelty_interval=int(args.novelty_interval), val_interval=int(args.test_accuracy_interval), diversity={'type': args.diversity_type, 'pdop': args.pairwise_diversity_op, 'ldop': args.layerwise_diversity_op, 'k': args.k, 'k_strat': args.k_strat}, scaled=scaled, rand_tech=args.rand_tech)
        helper.run(seed=False)
        helper.config['dataset'] = args.dataset.lower()
        helper.config['batch_size'] = args.batch_size
        helper.config['lr'] = args.lr
        helper.config['experiment_name'] = args.experiment_name
        helper.config['experiment_type'] = 'ae_unsup_train'
        helper.config['diversity_type'] = args.diversity_type
        helper.config['pairwise_diversity_op'] = args.pairwise_diversity_op
        helper.config['layerwise_diversity_op'] = args.layerwise_diversity_op
        helper.config['k'] = args.k
        helper.config['k_strat'] = args.k_strat
        helper.config['rand_tech'] = args.rand_tech
        helper.update_config()
            # for c in classlist:
            #     classwise_accuracy_record[run_num][i][np.where(classlist==c)[0][0]] = record_accuracy[c]

    # with open('output/' + experiment_name + '/classwise_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(classwise_accuracy_record,f)

if __name__ == '__main__':
    run()