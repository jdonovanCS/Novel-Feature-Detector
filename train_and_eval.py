import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import pickle
import os
import argparse
import random

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--dataset', help='which dataset should be used for training metric, choices are: cifar-10, cifar-100', default='random')
parser.add_argument('--experiment_name', help='experiment name for saving data related to training')
parser.add_argument('--fixed_conv', help='Should the convolutional layers stay fixed, or alternatively be trained', action='store_true')
parser.add_argument('--training_interval', help='How often should the network be trained. Values should be supplied as a fraction and will relate to the generations from evolution' +
'For example if 1 is given the filters generated from the final generation of evolution will be the only ones trained. If 0.5 is given then the halfway point of evolutionary generations and the final generation will be trained. ' +
'If 0 is given, the filters from every generation will be trained', type=float, default=1.)
parser.add_argument('--epochs', help="Number of epochos to train for", type=int, default=64)
parser.add_argument('--random', action='store_true')
parser.add_argument('--novelty_interval', help='How often should a novelty score be captured during training?', default=0)
parser.add_argument('--test_accuracy_interval', help='How often should test accuracy be assessed during training?', default=0)
parser.add_argument('--batch_size', help="batch size for training", type=int, default=64)
parser.add_argument('--evo_gens', help="number of generations used in evolving solutions", default=None)
parser.add_argument('--evo_pop_size', help='Number of individuals in population when evolving solutions', default=None)
parser.add_argument('--evo_dataset_for_novelty', help='Dataset used for novelty computation during evolution and training', default=None)
parser.add_argument('--evo', help='evolved solutions, should only be true if random is not set', action='store_true')
parser.add_argument('--evo_num_runs', help='Number of runs used in evolution', default=None)
parser.add_argument('--evo_tourney_size', help='Size of tournaments in evolutionary algorithm selection', default=None)
parser.add_argument('--evo_num_winners', help='Number of winners in tournament in evolutionary algorithm', default=None)
parser.add_argument('--evo_num_children', help='Number of children in evolutionary algorithm', default=None)
    
args = parser.parse_args()

def run():

    torch.multiprocessing.freeze_support()

    pickled_filters = {}
    
    experiment_name = args.experiment_name
    training_interval = args.training_interval
    fixed_conv = args.fixed_conv
    filename = ''
    if args.random:
        filename = 'output/' + experiment_name + '/random_gen_solutions.pickle'
    else:
        filename = "output/" + experiment_name + "/solutions_over_time.pickle"

    # get filters from pickle file
    with open(filename, 'rb') as f:
        pickled_filters = pickle.load(f)

    if args.random:
        random.shuffle(pickled_filters['random'][0])
        pickled_filters['random'] = np.array(pickled_filters['random'])

    
    helper.run(seed=False)


    # get loader for train and test images and classes
    data_module = helper.get_data_module(args.dataset, args.batch_size)
    data_module.prepare_data()
    data_module.setup()

    epochs = args.epochs
    
    helper.config['dataset'] = args.dataset.lower()
    helper.config['batch_size'] = args.batch_size
    helper.config['experiment_name'] = args.experiment_name
    helper.config['evo_gens'] = args.evo_gens
    helper.config['evo_pop_size'] = args.evo_pop_size
    helper.config['evo_dataset_for_novelty'] = args.evo_dataset_for_novelty
    helper.config['evo'] = args.evo or args.random == None or not args.random
    helper.config['evo_num_runs'] = args.evo_num_runs
    helper.config['evo_tourney_size'] = args.evo_tourney_size
    helper.config['evo_num_winners'] = args.evo_num_winners
    helper.config['evo_num_children'] = args.evo_num_children
    helper.config['experiment_type'] = 'training'
    helper.update_config()
    
    
    # run training and evaluation and record metrics in above variables
    # for each type of evolution ran
    for name in pickled_filters.keys():
        for run_num in range(len(pickled_filters[name])):
            # run_num = np.where(pickled_filters[name] == filters_list)[0][0]
            # for each generation train the solution output at that generation
            for i in range(len(pickled_filters[name][run_num])):
                # if we only want to train the solution from the final generation, continue
                if training_interval != 0 and i*1.0 not in [(len(pickled_filters[name][run_num])/(1/training_interval)*j)-1 for j in range(1, int(1/training_interval)+1)]:
                    continue
                
                # else train the network and collect the metrics
                save_path = "trained_models/trained/conv{}_e{}_n{}_r{}_g{}.pth".format(not fixed_conv, experiment_name, name, run_num, i)
                print('Training and Evaluating: {} Gen: {} Run: {}'.format(name, i, run_num))
                record_progress = helper.train_network(data_module=data_module, filters=pickled_filters[name][run_num][i], epochs=epochs, save_path=save_path, fixed_conv=fixed_conv, novelty_interval=int(args.novelty_interval), val_interval=int(args.test_accuracy_interval))
                helper.run(seed=False)
                helper.config['dataset'] = args.dataset.lower()
                helper.config['batch_size'] = args.batch_size
                helper.config['experiment_name'] = args.experiment_name
                helper.config['evo_gens'] = args.evo_gens
                helper.config['evo_pop_size'] = args.evo_pop_size
                helper.config['evo_dataset_for_novelty'] = args.evo_dataset_for_novelty
                helper.config['evo'] = args.evo or args.random == None or not args.random
                helper.config['evo_num_runs'] = args.evo_num_runs
                helper.config['evo_tourney_size'] = args.evo_tourney_size
                helper.config['evo_num_winners'] = args.evo_num_winners
                helper.config['evo_num_children'] = args.evo_num_children
                helper.config['experiment_type'] = 'training'
                helper.update_config()
                # for c in classlist:
                #     classwise_accuracy_record[name][run_num][i][np.where(classlist==c)[0][0]] = record_accuracy[c]

    # with open('output/' + experiment_name + '/classwise_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(classwise_accuracy_record,f)

if __name__ == '__main__':
    run()
