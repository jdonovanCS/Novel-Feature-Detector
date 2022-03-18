import pickle
import helper_hpc as helper
import torch
import numpy as np
import random
import os
import argparse

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--dataset', help='which dataset should be used for novelty metric, choices are: random, cifar-10', default='random')
parser.add_argument('--experiment_name', help='experiment name for saving data related to training', default='')
parser.add_argument('--fixed_conv', help='Should the convolutional layers stay fixed, or alternatively be trained', default=False)
parser.add_argument('--training_interval', help='How often should the network be trained. Values should be supplied as a fraction and will relate to the generations from evolution' +
'For example if 1 is given the filters generated from the final generation of evolution will be the only ones trained. If 0.5 is given then the halfway point of evolutionary generations and the final generation will be trained. ' +
'If 0 is given, the filters from every generation will be trained', default=1.)
args = parser.parse_args()

def save_final_accuracy_of_trained_models(pickle_path, save_path):

    # read in the saved training record
    with open(pickle_path, 'rb') as f:
        pickled_metrics = pickle.load(f)

    print(pickled_metrics['random'][0])
    modified_accuracies = {}
    for key in pickled_metrics:
        modified_accuracies[key] = np.zeros((len(pickled_metrics[key]), len(pickled_metrics[key][0][-1]['running_acc'])))
        for i in range(len(pickled_metrics[key])):
            modified_accuracies[key][i] = [p['accuracy'] for p in pickled_metrics[key][i][-1]['running_acc']]
    
    with open(save_path, 'wb') as f:
        pickle.dump(modified_accuracies, f)
    
    print(modified_accuracies)
    return modified_accuracies

# get the filters for random gens
def run():
    torch.multiprocessing.freeze_support()
    
    experiment_name = args.experiment_name
    trainin_interval = args.training_interval
    fixed_conv = True
    
    with open('output/' + experiment_name + '/random_gen_solutions.pickle', 'rb') as f:
        solutions = pickle.load(f)

    random.shuffle(solutions)
    solutions = np.array(solutions)
    num_to_train = 5
    name = 'random'

    helper.run()
    # experiment_name = "mutation_multiplier_small_edited_cifar100_pop20_gen50"

    
    # get loader for train and test images and classes
    trainset, testset, trainloader, testloader, classes = helper.load_CIFAR_10(helper.batch_size)
    
    # create variables for holding metric
    training_record = {}
    overall_accuracy_record = {}
    classwise_accuracy_record = {}
    classlist = np.array(classes)
    epochs = 1024


    overall_accuracy_record[name] = np.zeros((num_to_train, 1))
    classwise_accuracy_record[name] = np.zeros((num_to_train, 1, len(classlist)))
    training_record[name] = np.array([[dict for i in range(1)]for j in range(num_to_train)], dtype=dict)

    # for each filter in list of filters saved from randomly generated:
    for i in range (num_to_train):

        # train that network
        save_path = "trained_models/trained/conv{}_e{}_n{}_r{}.pth".format(not fixed_conv, experiment_name, name, i)
        print('Training and Evaluating: {} Run: {}'.format(name, i))
        record_progress = helper.train_network(trainloader=trainloader, filters=solutions[i], epochs=epochs, testloader=testloader, classes=classes, save_path=save_path, fixed_conv=fixed_conv)
        record_accuracy = helper.assess_accuracy(testloader=testloader, classes=classes, save_path=save_path)
        training_record[name][i][0] = record_progress
        overall_accuracy_record[name][i][0] = record_accuracy['overall']
        for c in classlist:
                    classwise_accuracy_record[name][i][0][np.where(classlist==c)[0][0]] = record_accuracy[c]


    # save as pickle files
    name_add = ''
    if fixed_conv: name_add = 'fixed_conv_'
    if not os.path.isdir('output/' + experiment_name):
        os.mkdir('output/' + experiment_name)
    with open('output/' + experiment_name + '/training_{}_{}over_time.pickle'.format(name, name_add), 'wb') as f:
        pickle.dump(training_record, f)
    with open('output/' + experiment_name + '/overall_accuracy_{}_{}over_time.pickle'.format(name, name_add), 'wb') as f:
        pickle.dump(overall_accuracy_record,f)
    with open('output/' + experiment_name + '/classwise_accuracy_{}_{}over_time.pickle'.format(name, name_add), 'wb') as f:
        pickle.dump(classwise_accuracy_record,f)
    # plot results
    cut_off_beginning = 0
    final_accuracies = save_final_accuracy_of_trained_models('output/' + experiment_name + '/training_{}_{}over_time.pickle'.format(name, name_add), 'output/' + experiment_name + '/final_accuracies_{}_{}over_training_time.pickle'.format(name, name_add))
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in overall_accuracy_record.items()], name=[k for k,x in overall_accuracy_record.items()], x_label="Generation", y_label="Fitness", compute_CI=True)
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in final_accuracies.items()], name=[k for k,x in final_accuracies.items()], x_label="Epoch", y_label="Accuracy", compute_CI=True)


if __name__ == "__main__":
    run()