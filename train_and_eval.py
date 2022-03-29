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
parser.add_argument('--batch_size', help="batch size for training", default=64)
parser.add_argument('--evo_gens', help="number of generations used in evolving solutions", default=None)
parser.add_argument('--evo_pop_size', help='Number of individuals in population when evolving solutions', default=None)
parser.add_argument('--evo_dataset_for_novelty', help='Dataset used for novelty computation during evolution and training', default=None)
parser.add_argument('--evo', help='evolved solutions, should only be true if random is not set', action='store_true')
parser.add_argument('--evo_num_runs', help='Number of runs used in evolution', default=None)
parser.add_argument('--evo_tourney_size', help='Size of tournaments in evolutionary algorithm selection', default=None)
parser.add_argument('--evo_num_winners', help='Number of winners in tournament in evolutionary algorithm', default=None)
parser.add_argument('--evo_num_children', help='Number of children in evolutionary algorithm', default=None)
    
args = parser.parse_args()

def save_final_accuracy_of_trained_models(pickle_path, save_path):

    # read in the saved training record
    with open(pickle_path, 'rb') as f:
        pickled_metrics = pickle.load(f)
    
    #     REMOVE
    #     for i, s in enumerate(pickled_metrics['random'][0]):
    #         if len(s['running_acc']) > 2:
    #             print(i)

    # numbers = [9, 19, 29, 39, 49]
    # for num in numbers:
    #     if round((((num+1)*1.0)/(50))*10) % (args.training_interval * 10) == 0:
    #         print(num)
    #     else:
    #         print(num)
    #         print(round((((num+1)*1.0)/(50))*10))
    #         print(round((((num+1)*1.0)/(50))*10) % (args.training_interval * 10))


    modified_accuracies = {}
    for key in pickled_metrics:
        modified_accuracies[key] = np.zeros((len(pickled_metrics[key])*int(1/args.training_interval), len(pickled_metrics[key][0][-1]['running_acc'])))
        for i in range(len(pickled_metrics[key])*int(1/args.training_interval)):
            if args.training_interval == 1:
                modified_accuracies[key][i] = [p['accuracy'] for p in pickled_metrics[key][i][int(len(pickled_metrics[key][i])/(int(1/args.training_interval))-1)]['running_acc']]
            elif args.random:
                modified_accuracies[key][i] = [p['accuracy'] for p in pickled_metrics[key][0][int(len(pickled_metrics[key][0])/(int(1/args.training_interval))*i-1)]['running_acc']]
            elif args.random != True and args.training_interval != 1:
                for j in range(int(1/args.training_interval)):
                    modified_accuracies[key][i*len(pickled_metrics[key])+j] = [p['accuracy'] for p in pickled_metrics[key][i][int(len(pickled_metrics[key][i])/(int(1/args.training_interval))*j-1)]['running_acc']]

    with open(save_path, 'wb') as f:
        pickle.dump(modified_accuracies, f)
    
    print(modified_accuracies)
    return modified_accuracies

def save_novelty_record_of_trained_models(pickle_path, save_path):

    # read in the saved training record
    with open(pickle_path, 'rb') as f:
        pickled_metrics = pickle.load(f)

    modified_novelties = {}
    for key in pickled_metrics:
        modified_novelties[key] = np.zeros((len(pickled_metrics[key])*int(1/args.training_interval), len(pickled_metrics[key][0][-1]['novelty_score'])))
        for i in range(len(pickled_metrics[key])*int(1/args.training_interval)):
            if args.training_interval == 1:
                modified_novelties[key][i] = [p['novelty'] for p in pickled_metrics[key][i][int(len(pickled_metrics[key][i])/(int(1/args.training_interval))-1)]['novelty_score']]
            elif args.random:
                modified_novelties[key][i] = [p['novelty'] for p in pickled_metrics[key][0][int(len(pickled_metrics[key][0])/(int(1/args.training_interval))*i-1)]['novelty_score']]
            elif args.random != True and args.training_interval != 1:
                for j in range(int(1/args.training_interval)):
                    modified_novelties[key][i*len(pickled_metrics[key])+j] = [p['novelty'] for p in pickled_metrics[key][i][int(len(pickled_metrics[key][i])/(int(1/args.training_interval))*j-1)]['novelty_score']]

    with open(save_path, 'wb') as f:
        import copy
        modified_novelties_for_file = copy.deepcopy(modified_novelties)
        modified_novelties_for_file['interval'] = args.novelty_interval
        pickle.dump(modified_novelties_for_file, f)
    
    print(modified_novelties)
    return modified_novelties

def save_final_test_accuracies_of_trained_models(pickle_path, save_path):
    # read in the saved training record
    with open(pickle_path, 'rb') as f:
        pickled_metrics = pickle.load(f)
        
    print(pickled_metrics[list(pickled_metrics.keys())[0]][0][-1]['test_accuracies'])    

    modified_test_accuracies = {}
    for key in pickled_metrics:
        modified_test_accuracies[key] = np.zeros((len(pickled_metrics[key])*int(1/args.training_interval), len(pickled_metrics[key][0][-1]['test_accuracies'])))
        for i in range(len(pickled_metrics[key])*int(1/args.training_interval)):
            if args.training_interval == 1:
                modified_test_accuracies[key][i] = [p['test_accuracy']['overall'] for p in pickled_metrics[key][i][int(len(pickled_metrics[key][i])/(int(1/args.training_interval))-1)]['test_accuracies']]
            elif args.random:
                modified_test_accuracies[key][i] = [p['test_accuracy']['overall'] for p in pickled_metrics[key][0][int(len(pickled_metrics[key][0])/(int(1/args.training_interval))*i-1)]['test_accuracies']]
            elif args.random != True and args.training_interval != 1:
                for j in range(int(1/args.training_interval)):
                    modified_test_accuracies[key][i*len(pickled_metrics[key])+j] = [p['test_accuracy']['overall'] for p in pickled_metrics[key][i][int(len(pickled_metrics[key][i])/(int(1/args.training_interval))*j-1)]['test_accuracies']]

    with open(save_path, 'wb') as f:
        import copy
        modified_test_accuracies_for_file = copy.deepcopy(modified_test_accuracies)
        modified_test_accuracies_for_file['interval'] = args.test_accuracy_interval
        pickle.dump(modified_test_accuracies_for_file, f)
    
    print(modified_test_accuracies)
    return modified_test_accuracies

def run():

    #REMOVE
    # name_add = ''
    # if args.random: name_add += 'random_'
    # if args.fixed_conv: name_add += 'fixed_conv_'
    # final_accuracies = save_final_accuracy_of_trained_models('output/' + args.experiment_name + '/training_{}over_time.pickle'.format(name_add), 'output/' + args.experiment_name + '/final_accuracies_{}over_training_time.pickle'.format(name_add))
    # final_novelties = save_novelty_record_of_trained_models('output/' + args.experiment_name + '/training_{}over_time.pickle'.format(name_add), 'output/' + args.experiment_name + '/novelty_{}over_training_time.pickle'.format(name_add))
    # final_test_accuracies = save_final_test_accuracies_of_trained_models('output/' + args.experiment_name + '/training_{}over_time.pickle'.format(name_add), 'output/' + args.experiment_name + '/final_test_accuracies_{}over_training_time.pickle'.format(name_add))
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[0:] for k, x in final_novelties.items()], name=[k for k,x in final_novelties.items()], x_label="Epoch", y_label="Novelty", compute_CI=True, show=True, sample_interval=4)
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[0:] for k, x in final_test_accuracies.items()], name=[k for k,x in final_test_accuracies.items()], x_label="Epoch", y_label="Accuracy", title="Final Test Accuracy on CIFAR-10, CIFAR-10 for novelty", compute_CI=True, show=True, sample_interval=4)
    # exit()

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

    
    helper.run(int(args.batch_size))


    # get loader for train and test images and classes
    data_module = helper.get_data_module(args.dataset, args.batch_size)
    # if args.dataset.lower() == 'cifar-100':
    #     trainset, testset, trainloader, testloader, classes = helper.load_CIFAR_100(helper.batch_size)
    # else:
    #     trainset, testset, trainloader, testloader, classes = helper.load_CIFAR_10(helper.batch_size)
    
    
    # create variables for holding metric
    # training_record = {}
    # overall_accuracy_record = {}
    # classwise_accuracy_record = {}
    # classlist = np.array(classes)
    epochs = args.epochs
    
    helper.wandb.config.dataset = args.dataset.lower()
    helper.wandb.config.batch_size = helper.batch_size
    helper.wandb.config.experiment_name = experiment_name
    helper.wandb.config.evo_gens = args.evo_gens
    helper.wandb.config.evo_pop = args.evo_pop
    helper.wandb.config.evo_dataset_for_novelty = args.evo_dataset_for_novelty
    helper.wandb.config.evo = args.evo or args.random == None or not args.random
    helper.wandb.config.evo_num_runs = args.evo_num_runs
    helper.wandb.config.evo_tourney_size = args.evo_tourney_size
    helper.wandb.config.evo_num_winners = args.evo_num_winners
    helper.wandb.config.evo_num_children = args.evo_num_children
    
    
    # run training and evaluation and record metrics in above variables
    # for each type of evolution ran
    for name in pickled_filters.keys():
        # instatiate entry in dictionary for this type of evolution
        # overall_accuracy_record[name] = np.zeros((len(pickled_filters[name]), len(pickled_filters[name][0])))
        # classwise_accuracy_record[name] = np.zeros((len(pickled_filters[name]), len(pickled_filters[name][0]), len(classlist)))
        # training_record[name] = np.array([[dict for i in range(len(pickled_filters[name][0]))]for j in range(len(pickled_filters[name]))], dtype=dict)
        # for each run of this evolution type
        for filters_list in pickled_filters[name]:
            run_num = np.where(pickled_filters[name] == filters_list)[0][0]
            # for each generation train the solution output at that generation
            for i in range (len(filters_list)):
                # if we only want to train the solution from the final generation
                # put zeros in the metric dictionaries if this isn't the final generation
                if training_interval != 0 and i*1.0 not in [(len(filters_list)/(1/training_interval)*j)-1 for j in range(1, int(1/training_interval)+1)]:
                    continue
                
                # else train the network and collect the metrics
                save_path = "trained_models/trained/conv{}_e{}_n{}_r{}_g{}.pth".format(not fixed_conv, experiment_name, name, run_num, i)
                print('Training and Evaluating: {} Gen: {} Run: {}'.format(name, i, run_num))
                record_progress = helper.train_network(data_module=data_module, filters=filters_list[i], epochs=epochs, save_path=save_path, fixed_conv=fixed_conv, novelty_interval=int(args.novelty_interval), val_interval=int(args.test_accuracy_interval))
                # record_accuracy = helper.assess_accuracy(testloader=testloader, classes=classes, save_path=save_path)
                # training_record[name][run_num][i] = record_progress
                # overall_accuracy_record[name][run_num][i] = record_accuracy['overall']
                # for c in classlist:
                #     classwise_accuracy_record[name][run_num][i][np.where(classlist==c)[0][0]] = record_accuracy[c]
    # name_add = ''
    # if args.random: name_add += 'random_'
    # if fixed_conv: name_add += 'fixed_conv_'
    # if not os.path.isdir('output/' + experiment_name):
    #     os.mkdir('output/' + experiment_name)
    # with open('output/' + experiment_name + '/training_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(training_record, f)
    # with open('output/' + experiment_name + '/overall_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(overall_accuracy_record,f)
    # with open('output/' + experiment_name + '/classwise_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(classwise_accuracy_record,f)

    # cut_off_beginning = 0
    # final_accuracies = save_final_accuracy_of_trained_models('output/' + experiment_name + '/training_{}over_time.pickle'.format(name_add), 'output/' + experiment_name + '/final_accuracies_{}over_training_time.pickle'.format(name_add))
    # final_novelties = save_novelty_record_of_trained_models('output/' + experiment_name + '/training_{}over_time.pickle'.format(name_add), 'output/' + experiment_name + '/novelty_{}over_training_time.pickle'.format(name_add))
    # final_test_accuracies = save_final_test_accuracies_of_trained_models('output/' + experiment_name + 'training_{}over_time.pickle'.format(name_add), 'output/' + experiment_name + '/final_test_accuracies {}over_training_time.pickle'.format(name_add))
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in overall_accuracy_record.items()], name=[k for k,x in overall_accuracy_record.items()], x_label="Generation", y_label="Fitness", compute_CI=True)
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in final_accuracies.items()], name=[k for k,x in final_accuracies.items()], x_label="Epoch", y_label="Accuracy", compute_CI=True)
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in final_novelties.items()], name=[k for k,x in final_novelties.items()], x_label="Epoch", y_label="Novelty", compute_CI=True, sample_interval=args.novelty_interval, save_path='output/' + experiment_name + 'novelty_{}over_training_time_plot'.format(name_add))
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in final_novelties.items()], name=[k for k,x in final_novelties.items()], x_label="Epoch", y_label="Accuracy", compute_CI=True, sample_interval=args.test_accuracy_interval, save_path='output/' + experiment_name + 'test_accuracy_{}over_training_time_plot'.format(name_add))



if __name__ == '__main__':
    run()
