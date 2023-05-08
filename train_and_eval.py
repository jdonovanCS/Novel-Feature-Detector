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
parser.add_argument('--fixed_conv', help='Should the convolutional layers stay fixed, or alternatively be trained', action='store_true')
parser.add_argument('--training_interval', help='How often should the network be trained. Values should be supplied as a fraction and will relate to the generations from evolution' +
'For example if 1 is given the filters generated from the final generation of evolution will be the only ones trained. If 0.5 is given then the halfway point of evolutionary generations and the final generation will be trained. ' +
'If 0 is given, the filters from every generation will be trained', type=float, default=1.)
parser.add_argument('--epochs', help="Number of epochos to train for", type=int, default=64)
parser.add_argument('--random', action='store_true')
parser.add_argument('--rand_norm', action='store_true')
parser.add_argument('--novelty_interval', help='How often should a novelty score be captured during training?', default=0)
parser.add_argument('--test_accuracy_interval', help='How often should test accuracy be assessed during training?', default=4)
parser.add_argument('--batch_size', help="batch size for training", type=int, default=64)
parser.add_argument('--lr', help='Learning rate for training', default=.001, type=float)
parser.add_argument('--evo_gens', help="number of generations used in evolving solutions", default=50)
parser.add_argument('--evo_pop_size', help='Number of individuals in population when evolving solutions', default=20)
parser.add_argument('--evo_dataset_for_novelty', help='Dataset used for novelty computation during evolution and training', default='cifar10')
parser.add_argument('--num_batches_for_evolution', help='Number of batches used of dataset when calculating diversity of filters', default=np.inf, type=int)
parser.add_argument('--evo', help='evolved solutions, should only be true if random is not set', action='store_true')
parser.add_argument('--gram-schmidt', help='gram-schmidt used to orthonormalize filters', action='store_true')
parser.add_argument('--ae_epochs', help='how many epochs were used in autoencoder unsupervised training', default=None, type=int)
parser.add_argument('--ae_dataset', help='dataset used for training autoencoder', default=None)
parser.add_argument('--ae_num_runs', help='number of runs for autoencoder training', default=None, type=int)
parser.add_argument('--unique_id', help='if a unique id is associated with the file the solution is stored in give it here.', default="", type=str)
parser.add_argument('--evo_num_runs', help='Number of runs used in evolution', default=5)
parser.add_argument('--evo_tourney_size', help='Size of tournaments in evolutionary algorithm selection', default=4)
parser.add_argument('--evo_num_winners', help='Number of winners in tournament in evolutionary algorithm', default=2)
parser.add_argument('--evo_num_children', help='Number of children in evolutionary algorithm', default=20)
parser.add_argument('--skip', default=0, help='skip the first n models to train, used mostly when a run fails partway through', type=int)
parser.add_argument('--stop_after', default=np.inf, help='stop after the first n models', type=int)
parser.add_argument('--diversity_type', type=str, default='absolute', help='Type of diversity metric to use for this experiment (ie. absolute, relative, original etc.)')
parser.add_argument('--num_workers', help='number of workers for training', default=np.inf, type=int)
args = parser.parse_args()

def run():

    torch.multiprocessing.freeze_support()

    stored_filters = {}
    
    experiment_name = args.experiment_name
    training_interval = args.training_interval
    fixed_conv = args.fixed_conv
    if args.random:
        name = 'random'
        if args.rand_norm:
            name = 'rand-normal'
    elif args.evo or (not args.random and not args.gram_schmidt):
        name = 'fitness'
    
    if args.gram_schmidt:
        name = 'gram-schmidt'
    if args.ae_epochs:
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

    if args.random or args.gram_schmidt:
        random.shuffle(stored_filters[0])
        stored_filters = np.array(stored_filters)
        with open(filename, 'wb') as f:
            np.save(f, stored_filters)
    
    helper.run(seed=False)


    # get loader for train and test images and classes
    data_module = helper.get_data_module(args.dataset, args.batch_size, args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    epochs = args.epochs
    
    helper.config['dataset'] = args.dataset.lower()
    helper.config['batch_size'] = args.batch_size
    helper.config['lr'] = args.lr
    helper.config['experiment_name'] = args.experiment_name
    helper.config['evo_gens'] = args.evo_gens
    helper.config['evo_pop_size'] = args.evo_pop_size
    helper.config['evo_dataset_for_novelty'] = args.evo_dataset_for_novelty
    helper.config['evo_num_batches_for_diversity'] = args.num_batches_for_evolution
    helper.config['evo'] = args.evo or (not args.random and not args.gram_schmidt)
    helper.config['evo_num_runs'] = args.evo_num_runs
    helper.config['evo_tourney_size'] = args.evo_tourney_size
    helper.config['evo_num_winners'] = args.evo_num_winners
    helper.config['evo_num_children'] = args.evo_num_children
    helper.config['experiment_type'] = 'training'
    helper.config['fixed_conv'] = fixed_conv == True
    helper.config['diversity_type'] = args.diversity_type
    helper.config['rand_norm'] = args.rand_norm
    helper.config['ae_epochs'] = args.ae_epochs
    helper.config['ae_dataset'] = args.ae_dataset
    helper.config['ae_num_runs'] = args.ae_num_runs
    helper.update_config()
    
    
    # run training and evaluation and record metrics in above variables
    # for each type of evolution ran

    if args.evo or (not args.random and not args.gram_schmidt): 
        skip = args.skip
        rand_skip=1
    else: 
        skip=0
        rand_skip = args.skip+1
    for run_num in range(len(stored_filters)):
        # run_num = np.where(stored_filters == filters_list)[0][0]
        # for each generation train the solution output at that generation
        for i in range(len(stored_filters[run_num])):
            # if we only want to train the solution from the final generation, continue
            if (training_interval != 0 and i*1.0 not in [(len(stored_filters[run_num])/(1/training_interval)*j)-1 for j in range(rand_skip, min(args.stop_after, int(1/training_interval)+1))]) or (training_interval==0 and i not in range(skip, args.stop_after)):
                continue
            
            # else train the network and collect the metrics
            helper.config['generation'] = i if (not args.random and not args.gram_schmidt) else None
            helper.update_config()
            save_path = "trained_models/trained/conv{}_e{}_n{}_r{}_g{}.pth".format(not fixed_conv, experiment_name, name, run_num, i)
            print('Training and Evaluating: {} Gen: {} Run: {}'.format(name, i, run_num))
            record_progress = helper.train_network(data_module=data_module, filters=stored_filters[run_num][i], epochs=epochs, lr=args.lr, save_path=save_path, fixed_conv=fixed_conv, novelty_interval=int(args.novelty_interval), val_interval=int(args.test_accuracy_interval), diversity_type=args.diversity_type)
            helper.run(seed=False)
            helper.config['dataset'] = args.dataset.lower()
            helper.config['batch_size'] = args.batch_size
            helper.config['lr'] = args.lr
            helper.config['experiment_name'] = args.experiment_name
            helper.config['evo_gens'] = args.evo_gens
            helper.config['evo_pop_size'] = args.evo_pop_size
            helper.config['evo_dataset_for_novelty'] = args.evo_dataset_for_novelty
            helper.config['evo_num_batches_for_diversity'] = args.num_batches_for_evolution
            helper.config['evo'] = args.evo or (not args.random and not args.gram_schmidt)
            helper.config['evo_num_runs'] = args.evo_num_runs
            helper.config['evo_tourney_size'] = args.evo_tourney_size
            helper.config['evo_num_winners'] = args.evo_num_winners
            helper.config['evo_num_children'] = args.evo_num_children
            helper.config['experiment_type'] = 'training'
            helper.config['fixed_conv'] = fixed_conv == True
            helper.config['rand_norm'] = args.rand_norm
            helper.config['ae_epochs'] = args.ae_epochs
            helper.config['ae_dataset'] = args.ae_dataset
            helper.config['ae_num_runs'] = args.ae_num_runs
            helper.update_config()
            # for c in classlist:
            #     classwise_accuracy_record[run_num][i][np.where(classlist==c)[0][0]] = record_accuracy[c]

    # with open('output/' + experiment_name + '/classwise_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(classwise_accuracy_record,f)

if __name__ == '__main__':
    run()
